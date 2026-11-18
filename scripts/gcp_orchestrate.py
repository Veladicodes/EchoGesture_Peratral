"""Provision a GCP VM, train + export remotely, pull artifacts, ALWAYS delete the VM (billing guard).

    python scripts/gcp_orchestrate.py --dry-run            # print every gcloud command, run nothing
    python scripts/gcp_orchestrate.py --profile auto       # try GPU options, fall back to CPU n2-standard-16

Flow: preflight (project, quotas, pre-existing VMs) -> create VM (GPU options, then CPU) -> bundle + scp ->
remote setup (uv + Python 3.11 venv + requirements) -> [optional remote extraction] -> scripts/train_pipeline.py
(background, polled) -> scp artifacts back -> `finally:` delete THIS run's VM and verify via `instances list`.
Only the instance created by this run is ever deleted; other VMs in the project are never touched.
scripts/ only: never imported by src/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GPU_IMAGE_FALLBACK = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
GPU_IMAGE_PROJECT = "deeplearning-platform-release"
GPU_OPTIONS = [  # tried in order; a quota/stock error moves on to the next
    {"zone": "us-central1-a", "machine": "g2-standard-8", "accel": None, "label": "1x NVIDIA L4"},
    {"zone": "us-east1-b", "machine": "g2-standard-8", "accel": None, "label": "1x NVIDIA L4"},
    {"zone": "us-west1-b", "machine": "g2-standard-8", "accel": None, "label": "1x NVIDIA L4"},
    {"zone": "us-central1-a", "machine": "n1-standard-8", "accel": "nvidia-tesla-t4", "label": "1x NVIDIA T4"},
    {"zone": "us-east1-c", "machine": "n1-standard-8", "accel": "nvidia-tesla-t4", "label": "1x NVIDIA T4"},
]
CPU_OPTION = {"zone": "us-central1-a", "machine": "n2-standard-16", "accel": None, "label": "CPU only (no GPU quota)",
              "image_family": "ubuntu-2204-lts", "image_project": "ubuntu-os-cloud"}

GPU_TRAIN_ARGS = "--ablation-epochs 40 --teacher-epochs 80 --student-epochs 120"
CPU_TRAIN_ARGS = "--ablation-epochs 15 --teacher-epochs 12 --student-epochs 40"  # CPU compute budget, see README


class GcloudError(RuntimeError):
    pass


class BillingGuardError(RuntimeError):
    pass


def real_gcloud(args: list[str], check: bool = True, capture: bool = False, timeout: int | None = None) -> str:
    exe = shutil.which("gcloud") or "gcloud"
    r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise GcloudError(f"gcloud {' '.join(args[:4])} failed ({r.returncode}): {r.stderr.strip()[-600:]}")
    return r.stdout if capture or True else ""


def dry_gcloud(args: list[str], check: bool = True, capture: bool = False, timeout: int | None = None) -> str:
    print("DRY-RUN gcloud " + " ".join(args))
    return "__DONE__ exit=0" if "remote_run.sh poll" in " ".join(args) else ""


def build_create_cmd(name: str, opt: dict, gpu_image: str = GPU_IMAGE_FALLBACK) -> list[str]:
    cmd = ["compute", "instances", "create", name, f"--zone={opt['zone']}", f"--machine-type={opt['machine']}",
           "--boot-disk-size=100GB", "--quiet"]
    if opt.get("image_family"):
        cmd += [f"--image-family={opt['image_family']}", f"--image-project={opt['image_project']}"]
    else:
        cmd += [f"--image-family={gpu_image}", f"--image-project={GPU_IMAGE_PROJECT}",
                "--maintenance-policy=TERMINATE", "--metadata=install-nvidia-driver=True"]
    if opt.get("accel"):
        cmd += [f"--accelerator=type={opt['accel']},count=1"]
    return cmd


def make_bundle(path: Path, include_cache: bool) -> Path:
    def keep(ti: tarfile.TarInfo):
        return None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti

    with tarfile.open(path, "w:gz") as t:
        for rel in ("config.yaml", "requirements.txt", "src", "scripts", "data/vocab.json",
                    "data/train_manifest.csv", "data/val_manifest.csv", "data/test_manifest.csv"):
            t.add(ROOT / rel, arcname=rel, filter=keep)
        for task in (ROOT / "weights").glob("*.task"):
            t.add(task, arcname=f"weights/{task.name}")
        if include_cache and (ROOT / "data/cache").is_dir():
            t.add(ROOT / "data/cache", arcname="data/cache")
    return path


SETUP = (
    "set -e; sudo apt-get update -y -qq; "
    "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip libgl1 libglib2.0-0 unzip tar; "
    "mkdir -p ~/echogesture && tar -xzf ~/bundle.tgz -C ~/echogesture && cd ~/echogesture && "
    "python3 -m pip install -q --user uv && export PATH=$HOME/.local/bin:$PATH && "
    "uv venv --python 3.11 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt && "
    ".venv/bin/python -c 'import torch,onnx,onnxruntime,mediapipe,tensorflow;print(\"deps ok\",torch.__version__,torch.cuda.is_available())'"
)


def orchestrate(run, profile_override: str | None = "auto", dry_pull: bool = False, poll_s: int = 120,
                max_hours: float = 5.0, remote_extract: bool = False, zip_path: str | None = None) -> dict:
    """profile_override: 'cpu' | 'gpu' | 'auto'/None (GPU options first, then CPU)."""
    name = "echogesture-train-" + datetime.now().strftime("%m%d-%H%M%S")
    info: dict = {"vm": name}
    created_zone: str | None = None
    try:
        # ---- preflight (read-only) ----
        info["project"] = run(["config", "get-value", "project"], check=False, capture=True).strip()
        before = run(["compute", "instances", "list", "--format=value(name)"], check=False, capture=True).split()
        info["pre_existing_vms_untouched"] = before
        # ---- create ----
        options = []
        if profile_override in (None, "auto", "gpu"):
            options += GPU_OPTIONS
        if profile_override in (None, "auto", "cpu"):
            options.append(CPU_OPTION)
        gpu_image = GPU_IMAGE_FALLBACK
        if profile_override != "cpu":
            img = run(["compute", "images", "list", f"--project={GPU_IMAGE_PROJECT}", "--no-standard-images",
                       "--filter=family~^pytorch-2-.*ubuntu-2204-nvidia", "--format=value(family)",
                       "--sort-by=~creationTimestamp", "--limit=1"], check=False, capture=True).strip()
            gpu_image = img or GPU_IMAGE_FALLBACK
        last_err = None
        for opt in options:
            try:
                run(build_create_cmd(name, opt, gpu_image))
                created_zone, info["option"] = opt["zone"], opt
                break
            except GcloudError as e:
                last_err = e
                print(f"create failed in {opt['zone']} ({opt['label']}): {str(e)[:200]}", flush=True)
                run(["compute", "instances", "delete", name, f"--zone={opt['zone']}", "--quiet"], check=False)
        if created_zone is None:
            raise GcloudError(f"could not create any VM: {last_err}")
        is_gpu = bool(info["option"].get("accel")) or info["option"]["machine"].startswith("g2-")
        train_args = GPU_TRAIN_ARGS if is_gpu else CPU_TRAIN_ARGS
        z = f"--zone={created_zone}"
        ssh = lambda cmd: run(["compute", "ssh", name, z, "--quiet", "--strict-host-key-checking=no", f"--command={cmd}"],  # noqa: E731
                              capture=True)
        # ---- wait for ssh, upload ----
        for _ in range(30):
            try:
                ssh("true")
                break
            except GcloudError:
                time.sleep(min(poll_s, 10))
        bundle = ROOT / "gcp_bundle.tgz"
        if not dry_pull:
            make_bundle(bundle, include_cache=not remote_extract)
        run(["compute", "scp", str(bundle), f"{name}:bundle.tgz", z, "--quiet", "--strict-host-key-checking=no"])
        if remote_extract and zip_path:
            run(["compute", "scp", zip_path, f"{name}:archive.zip", z, "--quiet", "--strict-host-key-checking=no"])
        ssh("mkdir -p ~/echogesture; tar -xzf ~/bundle.tgz -C ~/echogesture")
        remote = "bash ~/echogesture/scripts/remote_run.sh"
        ssh(f"{remote} setup")
        if remote_extract:
            ssh(f"{remote} extract")
        # ---- train (background, polled) ----
        ssh(f"{remote} train {train_args}")
        deadline = time.time() + max_hours * 3600
        while True:
            out = ssh(f"{remote} poll")
            print(out.strip().splitlines()[-1] if out.strip() else "(waiting)", flush=True)
            if "__DONE__" in out:
                info["remote_exit"] = out.split("exit=")[-1].strip()
                break
            if time.time() > deadline:
                raise TimeoutError(f"remote training exceeded {max_hours} h")
            time.sleep(poll_s)
        if info["remote_exit"] != "0":
            raise GcloudError(f"remote training failed (exit {info['remote_exit']}); see train_stdout.log")
        # ---- pull artifacts ----
        pulls = [("weights/teacher_best.pt", "weights/"), ("weights/student_best.pt", "weights/"),
                 ("weights/student_int8.onnx", "weights/"), ("weights/student_int8.tflite", "weights/"),
                 ("outputs/training_summary.json", "outputs/"), ("outputs/train_log.txt", "outputs/"),
                 ("train_stdout.log", "outputs/"), ("config.yaml", "")]
        for remote, local in pulls:
            run(["compute", "scp", f"{name}:echogesture/{remote}", str(ROOT / local) if local else str(ROOT / "config.yaml"),
                 z, "--quiet", "--strict-host-key-checking=no"])
        info["pulled"] = [r for r, _ in pulls]
        return info
    finally:
        _teardown(run, name, created_zone, info)


def _teardown(run, name: str, zone: str | None, info: dict) -> None:
    """Delete THIS run's VM (retry), then verify it is gone from `instances list`. Never touches other VMs."""
    if zone is not None:
        err = None
        for attempt in range(3):
            try:
                run(["compute", "instances", "delete", name, f"--zone={zone}", "--quiet"])
                err = None
                break
            except GcloudError as e:
                err = e
                time.sleep(2)
        if err is not None:
            raise BillingGuardError(f"FAILED TO DELETE VM {name}! Run manually: gcloud compute instances delete {name} "
                                    f"--zone={zone} --quiet  ({err})")
    listed = run(["compute", "instances", "list", "--format=value(name)"], check=False, capture=True).split()
    info["vms_remaining_in_project"] = listed
    if name in listed:
        raise BillingGuardError(f"VM {name} STILL LISTED after delete: {listed}")
    print(f"BILLING GUARD OK: {name} deleted and absent from instances list. Other VMs (not ours, untouched): {listed}",
          flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--poll-s", type=int, default=120)
    ap.add_argument("--remote-extract", action="store_true", help="upload the zip and run gcp_extract.py on the VM")
    ap.add_argument("--zip", default=r"F:\Samsung\archive (1).zip")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    info = orchestrate(dry_gcloud if a.dry_run else real_gcloud, a.profile, dry_pull=a.dry_run, poll_s=a.poll_s,
                       max_hours=a.max_hours, remote_extract=a.remote_extract, zip_path=a.zip)
    print(info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
