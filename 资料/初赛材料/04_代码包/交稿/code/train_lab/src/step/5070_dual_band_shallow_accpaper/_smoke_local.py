"""方案19 冒烟：参数量 + 前向 + 损失（不依赖预处理全量）。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "self_model"))

from model import DualBandShallowGate, count_params
from shallowfbcsp import ShallowFBCSPNet


def main() -> None:
    B, C, T = 4, 8, 500
    x_mu = torch.randn(B, C, T)
    x_beta = torch.randn(B, C, T)
    y = torch.randint(0, 3, (B,))

    v1 = ShallowFBCSPNet(C, 3, T, n_filters_time=40, n_filters_spat=40, attn=None)
    n_v1 = count_params(v1)
    v2 = DualBandShallowGate(C, T, 3, n_filters_branch=20, fuse="gate", lambda_aux=0.5)
    n_v2 = count_params(v2)
    print(f"V1(S0) params={n_v1}")
    print(f"V2(gate) params={n_v2}  ratio={n_v2 / max(n_v1, 1):.3f}")
    assert n_v2 < n_v1 * 1.25, "V2 参数不应远超 V1（查是否误用 T40×2）"

    parts = v2(x_mu, x_beta, return_parts=True)
    assert parts["logits_final"].shape == (B, 3)
    assert parts["alpha"].shape == (B, 1)
    loss = v2.loss(parts, y)
    loss.backward()
    print("forward+backward OK  loss=", float(loss.detach()))

    for fuse in ("fixed05", "concat"):
        m = DualBandShallowGate(C, T, 3, fuse=fuse, lambda_aux=0.5)
        p = m(x_mu, x_beta, return_parts=True)
        _ = m.loss(p, y)
        print(f"fuse={fuse} OK")

    print("SMOKE PASS")


if __name__ == "__main__":
    main()
