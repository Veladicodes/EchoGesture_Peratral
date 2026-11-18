"""Export gates (CLAUDE.md 5.4) on a random-weight student: <0.5 MB, cosine>0.99, drift<1e-2.

Calibration/test inputs here are SYNTHETIC skeleton sequences (no real data exists yet), so these
gates prove the export machinery, not trained-model quality. Re-run with real val data later.
"""
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from src import models as M
from src import quantize as Q
from src.common import load_vocab, seed_everything
from src.features import NUM_JOINTS, T_FRAMES

NC = 7
NUM_CLASSES = len(load_vocab())
MAX_MB = 0.5
FORBIDDEN_OPS = {"Einsum", "ScatterND", "ScatterElements", "Loop", "If", "NonZero", "Scan"}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    seed_everything(42)
    out = tmp_path_factory.mktemp("export")
    model = M.StudentSTGCN(NUM_CLASSES, NC).eval()
    calib = Q.synthetic_calibration(200, seed=42, num_channels=NC)
    test_x = Q.synthetic_calibration(64, seed=7, num_channels=NC)
    fused = Q.fuse_student(model)
    onnx_fp32 = out / "student_fp32.onnx"
    onnx_int8 = out / "student_int8.onnx"
    Q.export_onnx_fp32(fused, onnx_fp32, NC)
    Q.quantize_onnx_int8(onnx_fp32, onnx_int8, calib)
    return dict(model=model, fused=fused, fp32=onnx_fp32, int8=onnx_int8, test_x=test_x, calib=calib, dir=out)


def test_synthetic_calibration_is_valid():
    c = Q.synthetic_calibration(8, seed=1, num_channels=NC)
    assert c.shape == (8, NC, T_FRAMES, NUM_JOINTS) and c.dtype == np.float32 and np.isfinite(c).all()
    assert np.array_equal(c, Q.synthetic_calibration(8, seed=1, num_channels=NC))  # deterministic


def test_fusion_preserves_outputs(artifacts):
    x = torch.from_numpy(artifacts["test_x"][:8])
    with torch.no_grad():
        a, b = artifacts["model"](x), artifacts["fused"](x)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_fused_model_has_no_conv_adjacent_batchnorm(artifacts):
    names = [type(m).__name__ for m in artifacts["fused"].modules()]
    # only the stand-alone input BatchNorm (data_bn) may remain
    assert names.count("BatchNorm2d") == 1


def test_fp32_onnx_matches_torch(artifacts):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(artifacts["fp32"]), providers=["CPUExecutionProvider"])
    x = artifacts["test_x"][:4]
    ref = artifacts["fused"](torch.from_numpy(x)).detach().numpy()
    got = np.concatenate([sess.run(None, {"features": x[i : i + 1]})[0] for i in range(4)])
    assert np.abs(ref - got).max() < 1e-4


@pytest.mark.parametrize("key", ["fp32", "int8"])
def test_onnx_graph_has_only_export_safe_ops(artifacts, key):
    ops = {n.op_type for n in onnx.load(str(artifacts[key])).graph.node}
    assert not (ops & FORBIDDEN_OPS), ops & FORBIDDEN_OPS


def test_int8_onnx_is_actually_quantized(artifacts):
    ops = {n.op_type for n in onnx.load(str(artifacts["int8"])).graph.node}
    assert any(o.startswith("QLinear") or o in ("QuantizeLinear", "ConvInteger", "MatMulInteger") for o in ops), ops


def test_int8_onnx_size_under_budget(artifacts):
    mb = Path(artifacts["int8"]).stat().st_size / 1e6
    print(f"\nINT8 ONNX size: {mb:.4f} MB")
    assert mb < MAX_MB


def test_int8_vs_fp32_cosine_and_drift(artifacts):
    m = Q.compare_onnx(artifacts["fp32"], artifacts["int8"], artifacts["test_x"])
    print(f"\nONNX FP32-vs-INT8: {m}")
    assert m["cosine_mean"] > 0.99 and m["cosine_min"] > 0.99
    assert m["max_prob_drift"] < 1e-2
    assert 0.0 <= m["top1_agreement"] <= 1.0


def test_qat_finetune_runs_and_keeps_export_shape():
    seed_everything(42)
    model = M.StudentSTGCN(NUM_CLASSES, NC)
    fused = Q.fuse_student(model.eval())
    x = torch.from_numpy(Q.synthetic_calibration(16, seed=3, num_channels=NC))
    y = torch.randint(0, NUM_CLASSES, (16,))
    before = {k: v.clone() for k, v in fused.state_dict().items() if k.endswith("weight")}
    out = Q.qat_finetune(fused, [(x[:8], y[:8]), (x[8:], y[8:])], epochs=1, lr=1e-3)
    assert out(x[:2]).shape == (2, NUM_CLASSES) and torch.isfinite(out(x[:2])).all()
    changed = any(not torch.equal(before[k], v) for k, v in out.state_dict().items() if k in before)
    assert changed, "QAT fine-tune should update weights"


def test_qat_epoch_hook_is_called_once_per_epoch():
    seed_everything(42)
    fused = Q.fuse_student(M.StudentSTGCN(NUM_CLASSES, NC).eval())
    x = torch.from_numpy(Q.synthetic_calibration(8, seed=3, num_channels=NC))
    y = torch.randint(0, NUM_CLASSES, (8,))
    calls = []
    Q.qat_finetune(fused, [(x, y)], epochs=3, lr=1e-3, epoch_hook=calls.append)
    assert calls == [0, 1, 2]


def test_tflite_export_and_gates(artifacts):
    tf = pytest.importorskip("tensorflow")
    path = artifacts["dir"] / "student_int8.tflite"
    info = Q.export_tflite_int8(artifacts["fused"], path, artifacts["calib"], NC, test_x=artifacts["test_x"])
    mb = path.stat().st_size / 1e6
    print(f"\nINT8 TFLite size: {mb:.4f} MB; {info}")
    assert mb < MAX_MB
    assert info["tf_float_vs_torch_max_abs"] < 1e-3  # weight transplant is faithful
    assert info["cosine_mean"] > 0.99 and info["max_prob_drift"] < 1e-2
