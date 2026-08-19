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

    # U1 temporal predictor
    mu1 = MaskFutureDualExpert(
        n_times=1000,
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_decoder=True,
        predictor_temporal=True,
    )
    ou1 = mu1(mu1.make_mask(x), x_full=x, train_mode=True)
    assert ou1["z_pre_future"].shape == (2, 40)
    lu1, _ = compute_losses(
        ou1, y, x,
        lambda_cls=1, lambda_pred=1, lambda_sig=0.05, lambda_dec=0.2,
        cls_cur=True, cls_final=True, cls_future=False,
        use_sigreg=True, sigreg=SIGReg(),
    )
    lu1.backward()
    print("U1 temporal predictor OK", ou1["z_pre_future"].shape)

    # U2 spectral decoder
    mu2 = MaskFutureDualExpert(
        n_times=1000,
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_spectral_decoder=True,
    )
    ou2 = mu2(mu2.make_mask(x), x_full=x, train_mode=True)
    assert "band_hat" in ou2 and ou2["band_hat"].shape == (2, 2)
    assert "x_hat_future" not in ou2
    lu2, meta2 = compute_losses(
        ou2, y, x,
        lambda_cls=1, lambda_pred=1, lambda_sig=0.05, lambda_dec=0.2,
        cls_cur=True, cls_final=True, cls_future=False,
        use_sigreg=True, sigreg=SIGReg(),
    )
    assert "l_spec" in meta2 or "l_dec" in meta2
    lu2.backward()
    print("U2 spectral decoder OK", ou2["band_hat"].shape)

    # U3 gate entropy
    mu3 = MaskFutureDualExpert(
        n_times=1000,
        use_predictor=True,
        use_expert_future=True,
        use_gate=True,
        use_decoder=True,
        gate_entropy=True,
    )
    ou3 = mu3(mu3.make_mask(x), x_full=x, train_mode=True)
    assert "H_cur" in ou3 and "alpha" in ou3
    assert ou3["alpha"].shape == (2, 1)
    lu3, _ = compute_losses(
        ou3, y, x,
        lambda_cls=1, lambda_pred=1, lambda_sig=0.05, lambda_dec=0.2,
        cls_cur=True, cls_final=True, cls_future=False,
        use_sigreg=True, sigreg=SIGReg(),
    )
    lu3.backward()
    print("U3 gate entropy OK", float(ou3["H_cur"].detach().mean()))

    # U12 / U13 / U123：经 arms_registry 正式开关组合（不 import train_kfold，避免 shared_hparams 路径冲突）
    from arms_registry import ARMS, assert_u_arm_flags

    assert_u_arm_flags()
    print("U arm flag assert OK")

    def _smoke_arm(arm_id: str, *, expect_band: bool, expect_wave: bool, expect_H: bool):
        arm = ARMS[arm_id]
        m = MaskFutureDualExpert(
            n_times=1000,
            n_outputs=3,
            use_predictor=arm.use_predictor,
            use_expert_future=arm.use_expert_future,
            use_gate=arm.use_gate,
            use_decoder=arm.use_decoder,
            predictor_temporal=arm.predictor_temporal,
            use_spectral_decoder=arm.use_spectral_decoder,
            gate_entropy=arm.gate_entropy,
        )
        assert m.predictor_temporal == arm.predictor_temporal
        assert m.use_spectral_decoder == arm.use_spectral_decoder
        assert m.gate_entropy == arm.gate_entropy
        o = m(m.make_mask(x), x_full=x, train_mode=True)
        assert o["z_pre_future"].shape == (2, 40)
        assert ("band_hat" in o) is expect_band
        assert ("x_hat_future" in o) is expect_wave
        assert ("H_cur" in o) is expect_H
        if expect_band:
            assert o["band_hat"].shape == (2, 2)
        if expect_wave:
            assert o["x_hat_future"].shape == (2, 8, 400)
        if expect_H:
            assert m.gate[0].in_features == 82  # 2*D + 2
        else:
            assert m.gate[0].in_features == 80
        loss, meta = compute_losses(
            o, y, x,
            lambda_cls=1, lambda_pred=1, lambda_sig=0.05, lambda_dec=0.2,
            cls_cur=True, cls_final=True, cls_future=False,
            use_sigreg=True, sigreg=SIGReg(),
        )
        if expect_band:
            assert "l_spec" in meta or "l_dec" in meta
        loss.backward()
        print(
            f"{arm_id} combo OK",
            f"temporal={m.predictor_temporal}",
            f"spec={m.use_spectral_decoder}",
            f"Hgate={m.gate_entropy}",
        )

    _smoke_arm("U12", expect_band=True, expect_wave=False, expect_H=False)
    _smoke_arm("U13", expect_band=False, expect_wave=True, expect_H=True)
    _smoke_arm("U123", expect_band=True, expect_wave=False, expect_H=True)

    print("forward OK", {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})
    print("t_prime", m.t_prime, "i_vis", m.i_vis[0], "..", m.i_vis[-1], "i_fut", m.i_fut[0])


if __name__ == "__main__":
    main()
