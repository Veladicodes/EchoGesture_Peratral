"""Student (export-safe, INT8-bound) and Teacher (training-only) skeleton GCNs (CLAUDE.md 5.3).

Both consume (N, C, 64, 53) feature tensors from features.build_features and return (N, num_classes)
logits. The graph comes from the single PARENTS constant in features.py.

StudentSTGCN export rule: Conv2d 1x1, depthwise Conv2d 9x1, BatchNorm2d, ReLU, static adjacency
multiplication with torch.matmul against a registered buffer, plus a Sigmoid inside the SE stream
gate. Nothing data dependent. Layer order inside a block is chosen so that Conv+BN(+ReLU) are
adjacent modules and can be fused with torch.ao.quantization.fuse_modules.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .common import get_num_channels
from .features import NUM_JOINTS, PARENTS

N_HAND_JOINTS = 42  # joints 0..41; body joints are 42..52


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_adjacency(parents: list[int], lo: int = 0, hi: int | None = None) -> torch.Tensor:
    """Symmetric-normalised (A + I) over the sub-graph of joints [lo, hi) from the parent list."""
    hi = len(parents) if hi is None else hi
    n = hi - lo
    a = np.eye(n, dtype=np.float32)
    for v in range(lo, hi):
        p = parents[v]
        if p != v and lo <= p < hi:
            a[v - lo, p - lo] = 1.0
            a[p - lo, v - lo] = 1.0
    d = a.sum(axis=1)
    dinv = 1.0 / np.sqrt(d)
    return torch.from_numpy(a * dinv[:, None] * dinv[None, :])


# --------------------------------------------------------------------------- #
# Student
# --------------------------------------------------------------------------- #
class STBlock(nn.Module):
    """x@A -> 1x1 conv-BN-ReLU -> depthwise 9x1 conv-BN -> 1x1 conv-BN -> (+residual) -> ReLU."""

    def __init__(self, cin: int, cout: int, adjacency: torch.Tensor, stride: int = 1):
        super().__init__()
        self.register_buffer("A", adjacency.clone())
        self.gconv = nn.Conv2d(cin, cout, 1, bias=False)
        self.gbn = nn.BatchNorm2d(cout)
        self.grelu = nn.ReLU()
        self.dw = nn.Conv2d(cout, cout, (9, 1), stride=(stride, 1), padding=(4, 0), groups=cout, bias=False)
        self.dwbn = nn.BatchNorm2d(cout)
        self.pw = nn.Conv2d(cout, cout, 1, bias=False)
        self.pwbn = nn.BatchNorm2d(cout)
        if cin != cout or stride != 1:
            self.res_conv = nn.Conv2d(cin, cout, 1, stride=(stride, 1), bias=False)
            self.res_bn = nn.BatchNorm2d(cout)
        else:
            self.res_conv = None
            self.res_bn = None
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        agg = torch.matmul(x, self.A)  # (N,C,T,V) @ (V,V): static graph aggregation
        y = self.grelu(self.gbn(self.gconv(agg)))
        y = self.dwbn(self.dw(y))
        y = self.pwbn(self.pw(y))
        r = x if self.res_conv is None else self.res_bn(self.res_conv(x))
        return self.out_relu(y + r)


class StreamGate(nn.Module):
    """Squeeze-and-Excitation gate weighting hand (0-41) vs trajectory/body (42-52) features."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.c = channels
        self.fc1 = nn.Conv2d(2 * channels, hidden, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, 2 * channels, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hand = x[:, :, :, :N_HAND_JOINTS]
        body = x[:, :, :, N_HAND_JOINTS:]
        sh = hand.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)  # (N,C,1,1)
        sb = body.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        g = self.sig(self.fc2(self.relu(self.fc1(torch.cat([sh, sb], dim=1)))))
        return torch.cat([hand * g[:, : self.c], body * g[:, self.c :]], dim=3)


class StudentSTGCN(nn.Module):
    """~72K-parameter INT8-bound ST-GCN. Input (N, C, 64, 53) -> logits (N, num_classes)."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int | None = None,
        widths: tuple[int, ...] = (32, 48, 80, 104),
        strides: tuple[int, ...] = (1, 2, 2, 1),
        gate_hidden: int = 24,
    ):
        super().__init__()
        in_channels = get_num_channels() if in_channels is None else in_channels
        adj = build_adjacency(PARENTS)
        self.data_bn = nn.BatchNorm2d(in_channels)
        blocks, c = [], in_channels
        for w, s in zip(widths, strides):
            blocks.append(STBlock(c, w, adj, s))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.gate = StreamGate(c, gate_hidden)
        self.fc = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.data_bn(x))
        x = self.gate(x)
        x = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)  # global average pool
        return self.fc(x).flatten(1)


# --------------------------------------------------------------------------- #
# Teacher (training only; never exported or shipped)
# --------------------------------------------------------------------------- #
class _TLayer(nn.Module):
    """Full ST-GCN layer: graph aggregation, 1x1 spatial conv, full 9x1 temporal conv, residual."""

    def __init__(self, cin: int, cout: int, adjacency: torch.Tensor, stride: int = 1, drop: float = 0.1):
        super().__init__()
        self.register_buffer("A", adjacency.clone())
        self.sconv = nn.Conv2d(cin, cout, 1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.tconv = nn.Conv2d(cout, cout, (9, 1), stride=(stride, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout(drop)
        self.res = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=(stride, 1)), nn.BatchNorm2d(cout))
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.sconv(torch.matmul(x, self.A))))
        y = self.drop(self.bn2(self.tconv(y)))
        return self.act(y + self.res(x))


class WristFaceAttention(nn.Module):
    """Spatial attention: each wrist (body joints 9,10) attends over the face joints (0..4)."""

    def __init__(self, c: int):
        super().__init__()
        self.q, self.k, self.v = (nn.Conv2d(c, c, 1) for _ in range(3))
        self.scale = 1.0 / math.sqrt(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N,C,T,11)
        wrist, face = x[..., 9:11], x[..., 0:5]
        q = self.q(wrist).permute(0, 2, 3, 1)  # (N,T,2,C)
        k = self.k(face).permute(0, 2, 1, 3)  # (N,T,C,5)
        v = self.v(face).permute(0, 2, 3, 1)  # (N,T,5,C)
        att = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)  # (N,T,2,5)
        ctx = torch.matmul(att, v).permute(0, 3, 1, 2)  # (N,C,T,2)
        return torch.cat([x[..., :9], wrist + ctx], dim=3)


class DualStreamTeacherGCN(nn.Module):
    """4-layer dual-stream ST-GCN. Branch 1: 42 hand joints. Branch 2: 11 trajectory joints with
    wrist-to-face spatial attention. Branches fused by bidirectional cross-attention."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int | None = None,
        widths: tuple[int, ...] = (64, 128, 224, 320),
        strides: tuple[int, ...] = (1, 2, 2, 1),
        heads: int = 8,
    ):
        super().__init__()
        in_channels = get_num_channels() if in_channels is None else in_channels
        self.data_bn = nn.BatchNorm2d(in_channels)
        a_hand = build_adjacency(PARENTS, 0, N_HAND_JOINTS)
        a_body = build_adjacency(PARENTS, N_HAND_JOINTS, NUM_JOINTS)

        def branch(adj):
            layers, c = [], in_channels
            for w, s in zip(widths, strides):
                layers.append(_TLayer(c, w, adj, s))
                c = w
            return nn.Sequential(*layers)

        self.hand = branch(a_hand)
        self.body = branch(a_body)
        d = widths[-1]
        self.wrist_face = WristFaceAttention(d)
        self.h2b = nn.MultiheadAttention(d, heads, batch_first=True)
        self.b2h = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_h, self.ln_b = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.data_bn(x)
        h = self.hand(x[..., :N_HAND_JOINTS])
        b = self.wrist_face(self.body(x[..., N_HAND_JOINTS:]))
        ht = h.mean(dim=3).transpose(1, 2)  # (N,T',D)
        bt = b.mean(dim=3).transpose(1, 2)
        h2, _ = self.h2b(ht, bt, bt)
        b2, _ = self.b2h(bt, ht, ht)
        ht = self.ln_h(ht + h2)
        bt = self.ln_b(bt + b2)
        return self.head(torch.cat([ht.mean(dim=1), bt.mean(dim=1)], dim=1))
