"""本地冒烟：feat_index + 扰动 + B8/B2/B10 关键开关（无需真实数据）。"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

from feat_index import assert_default_map, assert_future_perturbation
from losses import compute_losses
from model import MaskFutureDualExpert
from sigreg import SIGReg


def main() -> None:
    assert_default_map()
    print("feat_index OK")

    m = MaskFutureDualExpert(
        n_times=1000,
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_decoder=True,
        mask_learnable=True,
    )
    ratio = assert_future_perturbation(
        m.encoder, i_vis=m.i_vis, i_fut=m.i_fut, n_times=1000
    )
    print(f"future-perturb OK ratio={ratio:.3f}")

    x = torch.randn(2, 8, 1000)
    xm = m.make_mask(x)
    assert not torch.allclose(xm[..., -400:], torch.zeros_like(xm[..., -400:]))
    out = m(xm, x_full=x, train_mode=True)
    y = torch.tensor([1, 2])
    loss, _ = compute_losses(
        out,
        y,
        x,
        lambda_cls=1,
        lambda_pred=1,
        lambda_sig=0.05,
        lambda_dec=0.2,
        cls_cur=True,
        cls_final=True,
        cls_future=False,
        use_sigreg=True,
        sigreg=SIGReg(),
    )
    loss.backward()
    assert m.mask_token.grad is not None, "B8 mask_token 应有梯度"
    print("B8 mask_token grad OK")

    # B2：target 可回传
    m2 = MaskFutureDualExpert(use_predictor=True)
    x2 = torch.randn(2, 8, 1000)
    xm2 = m2.make_mask(x2)
    out2 = m2(xm2, x_full=x2, no_grad_target=False, train_mode=True)
    assert out2.get("target_detached") is False
    loss2, _ = compute_losses(
        out2,
        y,
        x2,
        lambda_cls=0,
        lambda_pred=1,
        lambda_sig=0,
        lambda_dec=0,
        cls_cur=False,
        cls_final=False,
        cls_future=False,
        use_sigreg=False,
        sigreg=None,
    )
    loss2.backward()
    assert m2.encoder.conv_time.conv.weight.grad is not None
    print("B2 target backprop OK")

    # B10 EMA
    m3 = MaskFutureDualExpert(use_predictor=True)
    m3.init_ema_encoder()
    x3 = torch.randn(2, 8, 1000)
    out3 = m3(m3.make_mask(x3), x_full=x3, ema_target=True, train_mode=True)
    assert out3.get("target_detached") is True
    m3.update_ema_encoder()
    print("B10 EMA OK")

    # B5a α=1 → p_final is p_cur
    m4 = MaskFutureDualExpert(
        use_predictor=True, use_expert_future=True, fixed_alpha=1.0
    )
    out4 = m4(m4.make_mask(x), x_full=x, train_mode=True)
    assert out4["p_final"] is out4["p_cur"]
    print("B5a no-dup p_final OK")

    print("forward OK", {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})
    print("t_prime", m.t_prime, "i_vis", m.i_vis[0], "..", m.i_vis[-1], "i_fut", m.i_fut[0])


if __name__ == "__main__":
    main()
