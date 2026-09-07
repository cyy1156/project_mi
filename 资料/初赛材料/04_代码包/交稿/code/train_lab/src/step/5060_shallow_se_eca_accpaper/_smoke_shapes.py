"""Quick shape / import smoke for scheme 14."""
from __future__ import annotations

import torch

from attention_front import ECA1d, SE1d
from models import build_s0, build_shallow_eca, build_shallow_se, count_attn_params
from shared_hparams import OUT_ROOT_TAG, SHARED
from task_runner import run_baseline_main  # noqa: F401


def main() -> None:
    b, c, t, o = 4, 8, 500, 3
    x = torch.randn(b, c, t)
    for name, m in [
        ("S0", build_s0(c, t, o, 0.5)),
        ("SE", build_shallow_se(c, t, o, 0.5, reduction=2)),
        ("ECA", build_shallow_eca(c, t, o, 0.5, k_size=3)),
    ]:
        y = m(x)
        n_attn = count_attn_params(m)
        n_all = sum(p.numel() for p in m.parameters())
        print(f"{name}: out={tuple(y.shape)} attn_params={n_attn} total={n_all}")

    se, eca = SE1d(8, 2), ECA1d(8, 3)
    x0 = torch.randn(2, 8, 500)
    with torch.no_grad():
        d_se = (se(x0) - x0).abs().max().item()
        d_eca = (eca(x0) - x0).abs().max().item()
    print(f"init max|SE(x)-x|={d_se:.2e}  max|ECA(x)-x|={d_eca:.2e}")
    print("OUT=", OUT_ROOT_TAG, "batch=", SHARED.batch_train, SHARED.batch_eval)
    print("OK")


if __name__ == "__main__":
    main()
