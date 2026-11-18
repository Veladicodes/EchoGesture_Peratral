"""Billing-guard tests for scripts/gcp_orchestrate.py. gcloud is fully mocked: no real VM, no spend."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("gcp_orchestrate", Path(__file__).resolve().parents[1] / "scripts" / "gcp_orchestrate.py")
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)

FOREIGN = ["agenttraffic-lab", "pqc-vm"]


class FakeGcloud:
    """Records every command. `fail_on` makes the first command containing that token raise."""

    def __init__(self, fail_on=None, create_quota_errors=0, remaining_after_delete=False):
        self.cmds, self.fail_on, self.quota_errors = [], fail_on, create_quota_errors
        self.remaining = remaining_after_delete
        self.created = []

    def deleted(self):
        return any("instances delete" in " ".join(c) for c in self.cmds)

    def __call__(self, args, check=True, capture=False, timeout=None):
        self.cmds.append(list(args))
        flat = " ".join(args)
        if "instances create" in flat:
            if self.quota_errors > 0:
                self.quota_errors -= 1
                raise G.GcloudError("Quota 'GPUS_ALL_REGIONS' exceeded")
            self.created.append(args[args.index("create") + 1])
            return ""
        if "remote_run.sh poll" in flat:
            if self.fail_on and self.fail_on in flat:
                raise G.GcloudError("boom")
            return "__DONE__ exit=0"
        if "instances list" in flat:
            ours = [] if (self.deleted() and not self.remaining) else self.created
            return "\n".join(FOREIGN + ours)
        if self.fail_on and self.fail_on in flat:
            raise G.GcloudError(f"boom on {self.fail_on}")
        return ""


def _run(fake, **kw):
    return G.orchestrate(fake, profile_override="cpu", dry_pull=True, poll_s=0, **kw)


def test_vm_deleted_after_success():
    f = FakeGcloud()
    _run(f)
    deletes = [c for c in f.cmds if "instances delete" in " ".join(c)]
    assert len(deletes) == 1 and "--quiet" in deletes[0]
    assert f.created[0] in deletes[0]                         # only OUR instance is deleted


def test_vm_deleted_even_when_training_fails():
    f = FakeGcloud(fail_on="remote_run.sh train")
    with pytest.raises(G.GcloudError):
        _run(f)
    assert f.deleted(), "billing guard must delete in finally"


def test_vm_deleted_when_upload_fails():
    f = FakeGcloud(fail_on="scp")
    with pytest.raises(G.GcloudError):
        _run(f)
    assert f.deleted()


def test_never_deletes_foreign_instances():
    f = FakeGcloud()
    _run(f)
    flat = [" ".join(c) for c in f.cmds if "instances delete" in " ".join(c)]
    assert flat and all(not any(n in s for n in FOREIGN) for s in flat)


def test_gpu_quota_error_falls_back_to_next_option_then_cpu():
    f = FakeGcloud(create_quota_errors=len(G.GPU_OPTIONS))
    G.orchestrate(f, profile_override=None, dry_pull=True, poll_s=0)
    creates = [" ".join(c) for c in f.cmds if "instances create" in " ".join(c)]
    assert len(creates) == len(G.GPU_OPTIONS) + 1
    assert "accelerator" not in creates[-1] and "n2-standard-16" in creates[-1]


def test_verification_fails_loudly_if_vm_still_listed():
    f = FakeGcloud(remaining_after_delete=True)
    with pytest.raises(G.BillingGuardError):
        _run(f)


def test_boot_disk_and_machine_flags():
    flat = " ".join(G.build_create_cmd("vm1", G.CPU_OPTION))
    assert "--boot-disk-size=100GB" in flat and "--machine-type=n2-standard-16" in flat
