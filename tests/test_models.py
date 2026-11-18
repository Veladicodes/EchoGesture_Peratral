"""Tests for src/models.py (CLAUDE.md 5.3): export-safe StudentSTGCN + training-only teacher."""
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src import models as M
from src.common import load_vocab
from src.features import NUM_JOINTS, PARENTS, T_FRAMES

NUM_CLASSES = len(load_vocab())
SRC_MODELS = Path(M.__file__).read_text(encoding="utf-8")


@pytest.fixture(params=[6, 7])
def nc(request):
    return request.param


def test_adjacency_is_static_symmetric_normalised():
    A = M.build_adjacency(PARENTS)
    assert A.shape == (53, 53)
    assert torch.allclose(A, A.T, atol=1e-6)
    assert torch.all(A >= 0) and torch.isfinite(A).all()
    assert torch.all(torch.diagonal(A) > 0)  # self loops
    # bones exist: wrist(0)-middle MCP chain, shoulders
    assert A[9, 0] > 0 and A[47, 48] > 0
    # no cross-hand / hand-body edges
    assert torch.all(A[:21, 21:] == 0) and torch.all(A[:42, 42:] == 0)


def test_student_output_shape_and_finite(nc):
    m = M.StudentSTGCN(num_classes=NUM_CLASSES, in_channels=nc).eval()
    x = torch.randn(3, nc, T_FRAMES, NUM_JOINTS)
    y = m(x)
    assert y.shape == (3, NUM_CLASSES) and torch.isfinite(y).all()


def test_student_param_count_near_72k():
    m = M.StudentSTGCN(num_classes=NUM_CLASSES, in_channels=7)
    n = M.count_params(m)
    print(f"\nSTUDENT PARAMS (measured): {n}")
    assert 60_000 <= n <= 85_000, n


def test_student_uses_only_export_safe_modules():
    m = M.StudentSTGCN(num_classes=NUM_CLASSES, in_channels=7)
    allowed = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Sigmoid, nn.Identity)
    for name, mod in m.named_modules():
        if len(list(mod.children())) == 0:
            assert isinstance(mod, allowed), f"{name}: {type(mod).__name__} not allowed"
        if isinstance(mod, nn.Conv2d):
            ks = tuple(mod.kernel_size)
            assert ks in ((1, 1), (9, 1)), f"{name}: kernel {ks}"
            if ks == (9, 1):
                assert mod.groups == mod.in_channels == mod.out_channels, f"{name}: 9x1 must be depthwise"


def test_student_adjacency_is_buffer_not_parameter():
    m = M.StudentSTGCN(num_classes=NUM_CLASSES, in_channels=7)
    buf_names = [n for n, _ in m.named_buffers() if n.endswith("A")]
    par_names = [n for n, _ in m.named_parameters() if n.endswith("A")]
    assert buf_names and not par_names


def test_no_forbidden_ops_in_source():
    # strip comments/docstrings crudely: forbidden names may only appear in the explanatory docstring
    code = re.sub(r'""".*?"""', "", SRC_MODELS, flags=re.S)
    code = "\n".join(l.split("#")[0] for l in code.splitlines())
    student_code = code.split("class DualStreamTeacherGCN")[0]
    for bad in ("einsum", "scatter_add", "scatter(", "index_add", "nonzero", "torch.where", ".item()"):
        assert bad not in student_code, f"forbidden op in StudentSTGCN path: {bad}"


def test_student_is_deterministic_in_eval():
    torch.manual_seed(42)
    m = M.StudentSTGCN(num_classes=NUM_CLASSES, in_channels=7).eval()
    x = torch.randn(2, 7, T_FRAMES, NUM_JOINTS)
    assert torch.equal(m(x), m(x))


def test_student_traces_to_a_static_graph():
    m = M.StudentSTGCN(num_classes=NUM_CLASSES, in_channels=7).eval()
    x = torch.randn(1, 7, T_FRAMES, NUM_JOINTS)
    tr = torch.jit.trace(m, x)
    assert torch.allclose(tr(x), m(x), atol=1e-5)


def test_teacher_output_shape_and_size():
    t = M.DualStreamTeacherGCN(num_classes=NUM_CLASSES, in_channels=7).eval()
    n = M.count_params(t)
    print(f"\nTEACHER PARAMS (measured): {n}")
    y = t(torch.randn(2, 7, T_FRAMES, NUM_JOINTS))
    assert y.shape == (2, NUM_CLASSES) and torch.isfinite(y).all()
    assert n > 1_000_000  # training-only model, far larger than the student


def test_teacher_is_not_the_exported_model():
    assert M.count_params(M.DualStreamTeacherGCN(NUM_CLASSES, 7)) > 20 * M.count_params(
        M.StudentSTGCN(NUM_CLASSES, 7)
    )
