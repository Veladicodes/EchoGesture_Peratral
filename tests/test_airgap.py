"""Air-gap enforcement (CLAUDE.md 2.4): no network imports in src/, no socket use at runtime."""
import ast
import re
import socket
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# Top-level modules / dotted prefixes that must never be imported from src/.
BANNED = (
    "requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "socket", "ssl", "ftplib",
    "smtplib", "telnetlib", "websocket", "websockets", "huggingface_hub", "torch.hub",
    "torchvision.models",  # pretrained downloads
)

# Explicit whitelist: every entry needs a reason (CLAUDE.md 2.4).
WHITELIST = {
    "PIL": "Pillow renders HUD text locally; makes no network calls",
    "psutil": "read-only local network byte counter for the HUD; makes no network calls",
}

BANNED_TEXT = re.compile(r"torch\.hub|pretrained\s*=\s*True|urlopen|\bwget\b|\bcurl\b|pip\s+install")


def _src_files():
    return sorted(p for p in SRC.rglob("*.py"))


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            for a in node.names:  # `from torch import hub`
                yield f"{node.module}.{a.name}"


def _is_banned(mod: str) -> bool:
    return any(mod == b or mod.startswith(b + ".") for b in BANNED)


def test_src_exists():
    assert _src_files(), "src/ has no python files"


@pytest.mark.parametrize("path", _src_files(), ids=lambda p: p.name)
def test_no_banned_imports(path):
    bad = [m for m in _imported_modules(path) if _is_banned(m)]
    assert not bad, f"{path.name} imports banned network modules: {bad}"


@pytest.mark.parametrize("path", _src_files(), ids=lambda p: p.name)
def test_no_download_idioms(path):
    m = BANNED_TEXT.search(path.read_text(encoding="utf-8"))
    assert m is None, f"{path.name} contains download idiom: {m.group(0)!r}"


def test_whitelist_entries_have_reasons():
    assert all(isinstance(v, str) and len(v) > 10 for v in WHITELIST.values())


def test_detector_catches_a_banned_import(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("import requests\nfrom torch import hub\n", encoding="utf-8")
    mods = list(_imported_modules(f))
    assert any(_is_banned(m) for m in mods)


def test_feature_pipeline_opens_no_sockets(monkeypatch):
    """Runtime proof: any socket connect during feature extraction would raise."""

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket.socket, "connect_ex", boom)
    monkeypatch.setattr(socket, "create_connection", boom)

    from src import features as F

    out = F.build_features(np.random.default_rng(0).normal(size=(40, 53, 3)).astype(np.float32))
    assert np.isfinite(out).all()
