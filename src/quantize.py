"""Fusion, QAT fine-tune and INT8 export (ONNX + TFLite) for StudentSTGCN (project spec 5.4).

    python -m src.quantize --random-weights            # prove the export path before any training
    python -m src.quantize --checkpoint weights/student_best.pt --calib-npy data/calib.npy

Calibration uses real training tensors when --calib-npy is given; otherwise SYNTHETIC skeleton
sequences are used and the report says so (no real data exists before the starter kit arrives).

ONNX : torch.onnx.export (FP32, BN folded) -> onnxruntime.quantization.quantize_static (INT8, QOperator).
TFLite: ai-edge-torch is Linux-first, so on Windows the fused weights are transplanted into an
        equivalent TensorFlow graph (NHWC) and converted with the full-integer TFLiteConverter.
        The transplant is verified against PyTorch (tf_float_vs_torch_max_abs) before conversion.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .common import load_config, load_vocab, repo_path, seed_everything
from .features import NUM_CHANNELS, NUM_JOINTS, T_FRAMES, build_features
from .models import N_HAND_JOINTS, StudentSTGCN, count_params

# --------------------------------------------------------------------------- #
# synthetic data (only used when no real calibration tensors exist)
# --------------------------------------------------------------------------- #
def _synthetic_joints(rng: np.random.Generator) -> np.ndarray:
    t = int(rng.integers(40, 120))
    ts = np.linspace(0, 1, t, dtype=np.float32)[:, None, None]
    j = np.zeros((t, NUM_JOINTS, 3), np.float32)
    for sl in (slice(0, 21), slice(21, 42)):
        shape = rng.normal(0, 0.03, (1, 21, 3)).astype(np.float32)
        shape[:, 0] = 0
        shape[:, 9] = [0.0, -0.08, 0.0]
        wiggle = 0.02 * np.sin(2 * np.pi * (rng.uniform(0.5, 3) * ts + rng.uniform(0, 1, (1, 21, 3))))
        walk = np.cumsum(rng.normal(0, 0.004, (t, 1, 3)), axis=0).astype(np.float32)
        centre = rng.uniform(0.3, 0.7, (1, 1, 3)).astype(np.float32)
        hand = centre + walk + shape + wiggle.astype(np.float32)
        present = np.ones(t, bool)
        if rng.random() < 0.3:  # a hand that leaves the frame for a stretch
            a = int(rng.integers(0, t - 5))
            present[a : a + int(rng.integers(3, 30))] = False
        j[:, sl] = np.where(present[:, None, None], hand, 0.0)
    body = np.array(
        [[0.5, 0.3, 0], [0.48, 0.28, 0], [0.52, 0.28, 0], [0.45, 0.3, 0], [0.55, 0.3, 0],
         [0.4, 0.5, 0], [0.6, 0.5, 0], [0.35, 0.7, 0], [0.65, 0.7, 0], [0.33, 0.85, 0], [0.67, 0.85, 0]],
        np.float32,
    )
    j[:, 42:] = body + np.cumsum(rng.normal(0, 0.002, (t, 11, 3)), axis=0).astype(np.float32)
    return j


def synthetic_calibration(n: int = 200, seed: int = 42, num_channels: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nc = NUM_CHANNELS if num_channels is None else num_channels
    return np.stack([build_features(_synthetic_joints(rng), nc) for _ in range(n)]).astype(np.float32)


# --------------------------------------------------------------------------- #
# fusion + QAT
# --------------------------------------------------------------------------- #
def fuse_student(model: StudentSTGCN) -> StudentSTGCN:
    """Fold Conv2d+BatchNorm2d(+ReLU) in every block (eval mode); data_bn stays a stand-alone BN."""
    import torch.ao.quantization as tq

    m = copy.deepcopy(model).eval()
    for blk in m.blocks:
        groups = [["gconv", "gbn", "grelu"], ["dw", "dwbn"], ["pw", "pwbn"]]
        if blk.res_conv is not None:
            groups.append(["res_conv", "res_bn"])
        tq.fuse_modules(blk, groups, inplace=True)
    return m


def _conv_of(mod: nn.Module) -> nn.Conv2d:
    return mod[0] if isinstance(mod, nn.Sequential) else mod  # fused ConvReLU2d is a Sequential


def qat_finetune(
    fused: StudentSTGCN,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 15,
    lr: float = 1e-4,
    epoch_hook=None,
) -> StudentSTGCN:
    """Fake-quant fine-tune the fused model, then copy the tuned float weights back.

    The returned model is a plain FP32 fused model whose weights were trained to tolerate INT8
    rounding; it is what gets exported/quantized. `batches` must be re-iterable (list or loader).
    """
    import torch.ao.quantization as tq

    qm = copy.deepcopy(fused).train()
    qm.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(qm, inplace=True)
    opt = torch.optim.Adam(qm.parameters(), lr=lr)
    for ep in range(epochs):
        if epoch_hook is not None:  # e.g. dataset.set_epoch so online augmentation differs every epoch
            epoch_hook(ep)
        for x, y in batches:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(qm(x), y)
            loss.backward()
            opt.step()

    out = copy.deepcopy(fused)
    targets = dict(out.named_modules())
    with torch.no_grad():
        for name, mod in qm.named_modules():
            if isinstance(mod, nn.Conv2d) and name in targets:
                dst = _conv_of(targets[name])
                dst.weight.copy_(mod.weight)
                if mod.bias is not None and dst.bias is not None:
                    dst.bias.copy_(mod.bias)
            elif isinstance(mod, nn.BatchNorm2d) and name in targets and isinstance(targets[name], nn.BatchNorm2d):
                targets[name].load_state_dict(mod.state_dict(), strict=False)
    return out.eval()


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #
def export_onnx_fp32(fused: nn.Module, path: str | Path, num_channels: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, num_channels, T_FRAMES, NUM_JOINTS)
    torch.onnx.export(
        fused.eval(), x, str(path), input_names=["features"], output_names=["logits"],
        opset_version=13, do_constant_folding=True, dynamo=False,
    )
    _inline_identity_initializers(path)
    return path


def _inline_identity_initializers(path: Path) -> None:
    """The exporter de-duplicates identical tensors (e.g. all-zero folded biases) behind Identity
    nodes; ORT's quantizer needs real initializers, so copy them back in and drop those nodes."""
    import onnx

    model = onnx.load(str(path))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    keep, changed = [], False
    for n in g.node:
        if n.op_type == "Identity" and n.input[0] in inits:
            t = onnx.TensorProto()
            t.CopyFrom(inits[n.input[0]])
            t.name = n.output[0]
            g.initializer.append(t)
            changed = True
        else:
            keep.append(n)
    if changed:
        del g.node[:]
        g.node.extend(keep)
        onnx.save(model, str(path))


def quantize_onnx_int8(fp32_path: str | Path, out_path: str | Path, calib: np.ndarray, method: str = "percentile",
                       percentile: float = 99.999, nodes_to_exclude: list[str] | None = None) -> Path:
    """Static INT8 quantization with representative tensors (calib: (N,C,64,53)).

    method: "percentile" | "entropy" | "minmax". nodes_to_exclude keeps named nodes in FP32 (e.g. the classifier
    "/fc/Conv"; the file stays far below 0.5 MB because only 104x100 weights stay float)."""
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self, arr: np.ndarray):
            self._it = iter(arr)

        def get_next(self):
            x = next(self._it, None)
            return None if x is None else {"features": x[None].astype(np.float32)}

    from onnxruntime.quantization.shape_inference import quant_pre_process

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pre_path = out_path.with_name(out_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)  # constants -> initializers
    # Percentile calibration, not MinMax: on the trained student MinMax collapsed val top-1 to ~1% (outlier
    # activations stretch the ranges), percentile (99.999) kept it within 0.3 pt of FP32 (measured, see report).
    cm = {"percentile": CalibrationMethod.Percentile, "entropy": CalibrationMethod.Entropy,
          "minmax": CalibrationMethod.MinMax}[method]
    extra = {"CalibPercentile": percentile} if method == "percentile" else {}
    quantize_static(
        str(pre_path), str(out_path), _Reader(calib),
        quant_format=QuantFormat.QOperator, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, per_channel=True,
        calibrate_method=cm, extra_options=extra, nodes_to_exclude=list(nodes_to_exclude or []),
    )
    pre_path.unlink(missing_ok=True)
    return out_path


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict:
    """ref/got: (N, classes) logits. Cosine on raw logits, plus mean-centred cosine (stricter)."""
    def cos(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)

    rc, gc = ref - ref.mean(-1, keepdims=True), got - got.mean(-1, keepdims=True)
    c = cos(ref, got)
    return {
        "cosine_mean": float(c.mean()),
        "cosine_min": float(c.min()),
        "cosine_centered_mean": float(cos(rc, gc).mean()),
        "max_logit_drift": float(np.abs(ref - got).max()),
        "max_prob_drift": float(np.abs(_softmax(ref) - _softmax(got)).max()),
        "top1_agreement": float((ref.argmax(-1) == got.argmax(-1)).mean()),
        "n": int(ref.shape[0]),
    }


def _run_onnx(path: str | Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"features": x[i : i + 1]})[0] for i in range(len(x))])


def compare_onnx(fp32_path, int8_path, x: np.ndarray) -> dict:
    return _metrics(_run_onnx(fp32_path, x), _run_onnx(int8_path, x))


# --------------------------------------------------------------------------- #
# TFLite (weights transplanted into an equivalent NHWC TensorFlow graph)
# --------------------------------------------------------------------------- #
def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def _build_tf_module(fused: StudentSTGCN, num_channels: int):
    import tensorflow as tf

    bn = fused.data_bn
    scale = _np(bn.weight) / np.sqrt(_np(bn.running_var) + bn.eps)
    shift = _np(bn.bias) - _np(bn.running_mean) * scale

    def k1(conv):  # torch (out,in,1,1) -> tf (1,1,in,out)
        return tf.constant(_np(conv.weight)[:, :, 0, 0].T[None, None]), tf.constant(_np(conv.bias))

    blocks = []
    for blk in fused.blocks:
        dw = blk.dw
        res = blk.res_conv
        blocks.append(dict(
            A=tf.constant(_np(blk.A)),
            g=k1(_conv_of(blk.gconv)),
            dw=(tf.constant(_np(dw.weight)[:, 0, :, 0].T[:, None, :, None]), tf.constant(_np(dw.bias))),
            stride=int(dw.stride[0]),
            pw=k1(blk.pw),
            res=None if res is None else k1(res),
        ))
    g = fused.gate
    c = g.c
    w1, b1 = tf.constant(_np(g.fc1.weight)[:, :, 0, 0].T), tf.constant(_np(g.fc1.bias))
    w2, b2 = tf.constant(_np(g.fc2.weight)[:, :, 0, 0].T), tf.constant(_np(g.fc2.bias))
    wf, bf = tf.constant(_np(fused.fc.weight)[:, :, 0, 0].T), tf.constant(_np(fused.fc.bias))
    sc, sh = tf.constant(scale), tf.constant(shift)

    class _Student(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, T_FRAMES, NUM_JOINTS, num_channels], tf.float32, name="features")])
        def __call__(self, x):  # x: (N,T,V,C)
            x = x * sc + sh
            for b in blocks:
                a = tf.transpose(tf.matmul(tf.transpose(x, [0, 1, 3, 2]), b["A"]), [0, 1, 3, 2])
                y = tf.nn.relu(tf.nn.conv2d(a, b["g"][0], 1, "VALID") + b["g"][1])
                y = tf.pad(y, [[0, 0], [4, 4], [0, 0], [0, 0]])
                # stride-1 depthwise then subsample: identical to a strided VALID conv, but the Linux TF CPU kernel
                # rejects unequal row/column strides.
                y = tf.nn.depthwise_conv2d(y, b["dw"][0], [1, 1, 1, 1], "VALID")[:, :: b["stride"], :, :] + b["dw"][1]
                y = tf.nn.conv2d(y, b["pw"][0], 1, "VALID") + b["pw"][1]
                r = x if b["res"] is None else tf.nn.conv2d(x, b["res"][0], [1, b["stride"], 1, 1], "VALID") + b["res"][1]
                x = tf.nn.relu(y + r)
            hand, body = x[:, :, :N_HAND_JOINTS, :], x[:, :, N_HAND_JOINTS:, :]
            s = tf.concat([tf.reduce_mean(hand, axis=[1, 2]), tf.reduce_mean(body, axis=[1, 2])], axis=-1)
            gate = tf.sigmoid(tf.matmul(tf.nn.relu(tf.matmul(s, w1) + b1), w2) + b2)
            x = tf.concat([hand * gate[:, None, None, :c], body * gate[:, None, None, c:]], axis=2)
            return tf.matmul(tf.reduce_mean(x, axis=[1, 2]), wf) + bf

    return tf, _Student()


def export_tflite_int8(
    fused: StudentSTGCN, path: str | Path, calib: np.ndarray, num_channels: int, test_x: np.ndarray | None = None,
    val_x: np.ndarray | None = None, val_y: np.ndarray | None = None,
) -> dict:
    """Full-integer TFLite (float32 I/O boundary, int8 inside). Input layout is NHWC (1,64,53,C)."""
    tf, mod = _build_tf_module(fused, num_channels)
    nhwc = lambda a: np.transpose(a, (0, 2, 3, 1)).astype(np.float32)  # (N,C,T,V)->(N,T,V,C)  # noqa: E731
    probe = (calib if test_x is None else test_x)[:8]
    with torch.no_grad():
        ref = fused(torch.from_numpy(probe)).numpy()
    tf_out = np.concatenate([mod(tf.constant(nhwc(probe[i : i + 1]))).numpy() for i in range(len(probe))])
    transplant_err = float(np.abs(ref - tf_out).max())

    cf = mod.__call__.get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf], mod)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative():
        for c in calib:
            yield [nhwc(c[None])]  # one list entry per model input

    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    blob = conv.convert()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)

    x = calib[:64] if test_x is None else test_x
    with torch.no_grad():
        ref = fused(torch.from_numpy(x)).numpy()
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    got = []
    for i in range(len(x)):
        it.set_tensor(inp["index"], nhwc(x[i : i + 1]))
        it.invoke()
        got.append(it.get_tensor(outp["index"]).copy())
    info = _metrics(ref, np.concatenate(got))
    if val_x is not None and val_y is not None:  # real accuracy of the INT8 TFLite and of its FP32 source
        preds = []
        for i in range(len(val_x)):
            it.set_tensor(inp["index"], nhwc(val_x[i : i + 1]))
            it.invoke()
            preds.append(int(it.get_tensor(outp["index"]).argmax()))
        with torch.no_grad():
            fp = fused(torch.from_numpy(val_x)).argmax(1).numpy()
        info["val_top1_int8_tflite"] = float((np.array(preds) == val_y).mean())
        info["val_top1_fp32"] = float((fp == val_y).mean())
        info["val_n"] = int(len(val_y))
    info.update(
        size_mb=path.stat().st_size / 1e6,
        tf_float_vs_torch_max_abs=transplant_err,
        input_layout="NHWC (1,64,53,C)",
        input_dtype=str(np.dtype(inp["dtype"])),
    )
    return info


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> dict:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Fuse + export StudentSTGCN to INT8 ONNX/TFLite")
    ap.add_argument("--checkpoint", help="student state_dict (.pt)")
    ap.add_argument("--random-weights", action="store_true", help="export an untrained student (pipeline proof)")
    ap.add_argument("--calib-npy", help="real calibration tensors (N,C,64,53)")
    ap.add_argument("--calib_samples", type=int, default=int(cfg["calib_samples"]))
    ap.add_argument("--skip-tflite", action="store_true")
    ap.add_argument("--out-dir", default=cfg["paths"]["weights"])
    args = ap.parse_args(argv)
    if not args.checkpoint and not args.random_weights:
        ap.error("pass --checkpoint or --random-weights")

    seed_everything(42)
    nc, ncls = NUM_CHANNELS, len(load_vocab())
    model = StudentSTGCN(ncls, nc)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    if args.calib_npy:
        calib, calib_src = np.load(args.calib_npy).astype(np.float32)[: args.calib_samples], "real"
    else:
        calib, calib_src = synthetic_calibration(args.calib_samples, 42, nc), "SYNTHETIC (no real data yet)"
    test_x = synthetic_calibration(64, 7, nc) if not args.calib_npy else calib[-64:]

    out = repo_path(args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    fused = fuse_student(model)
    fp32 = export_onnx_fp32(fused, out / "student_fp32.onnx", nc)
    int8 = quantize_onnx_int8(fp32, out / cfg["paths"]["student_onnx"].split("/")[-1], calib)
    report = {
        "weights": "random (untrained)" if args.random_weights else args.checkpoint,
        "calibration_source": calib_src, "calib_samples": int(len(calib)),
        "num_channels": nc, "num_classes": ncls, "student_params": count_params(model),
        "onnx_int8_mb": int8.stat().st_size / 1e6, "onnx": compare_onnx(fp32, int8, test_x),
    }
    if not args.skip_tflite:
        try:
            report["tflite"] = export_tflite_int8(
                fused, out / cfg["paths"]["student_tflite"].split("/")[-1], calib, nc, test_x
            )
        except ImportError as e:  # project spec 2.5: say so, never skip silently
            report["tflite"] = f"NOT EXPORTED on this machine: {e}. Run in Docker/WSL/GCP Linux."
    (repo_path(cfg["paths"]["outputs"]) / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
