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

    # ---- T 系列 v3：E_pos token + token L_pred + AttnPool（无 Cross-Attn / 无 Phase）----
    from arms_registry import assert_t_arm_flags

    assert_t_arm_flags()
    print("T arm flag assert OK")

    def _smoke_t(arm_id: str, *, embed: int = 40):
        arm = ARMS[arm_id]
        m = MaskFutureDualExpert(
            n_times=1000,
            n_outputs=3,
            embed_dim=int(arm.extra.get("embed_dim", embed)),
            use_predictor=arm.use_predictor,
            use_expert_future=arm.use_expert_future,
            use_gate=arm.use_gate,
            use_decoder=arm.use_decoder,
            predictor_pos_token=arm.predictor_pos_token,
            pred_token_seq=arm.pred_token_seq,
            expert_attn_pool=arm.expert_attn_pool,
        )
        assert m.predictor_pos_token and m.pred_token_seq and m.expert_attn_pool
        n_fut = len(m.i_fut)
        o = m(m.make_mask(x), x_full=x, train_mode=True)
        assert "z_pre_future_seq" in o
        assert o["z_pre_future_seq"].shape == (2, n_fut, m.embed_dim)
        assert o["z_pre_future"].shape == (2, m.embed_dim)
        assert "z_target_future_seq" in o
        assert o["z_target_future_seq"].shape == (2, n_fut, m.embed_dim)
        assert "phase_ids" not in o
        assert "phase_logits" not in o
        assert "x_hat_future" in o and o["x_hat_future"].shape == (2, 8, 400)
        loss, meta = compute_losses(
            o,
            torch.tensor([1, 0], dtype=torch.long),
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
            lambda_phase=0.0,
        )
        assert "l_pred_token" in meta
        loss.backward()
        print(
            f"{arm_id} OK",
            f"D={m.embed_dim}",
            f"L_fut={n_fut}",
            f"l_pred={meta.get('l_pred'):.4f}",
        )

    _smoke_t("T1")
    _smoke_t("T1_128", embed=128)

    # ---- 方案 21：F_mi / pf800 / A2_pt 结构 / J1 in-window JEPA ----
    from arms_registry import ARMS, assert_21_arm_flags
    from inwin_jepa import sample_block_starts
    from scheme21_data import crop_pf_mi080, filter_indices_by_t0

    assert_21_arm_flags()
    print("21 arm flag assert OK")

    t0 = __import__("numpy").array([0.4, 1.0, 1.5, 2.0], dtype=__import__("numpy").float32)
    idx = __import__("numpy").arange(4)
    fidx = filter_indices_by_t0(idx, t0, t0_max=1.0)
    assert fidx.tolist() == [0, 1], f"F_mi_a filter got {fidx.tolist()}"
    print("scheme21 t0 filter OK", fidx.tolist())

    x1k = torch.randn(2, 8, 1000)
    x800 = crop_pf_mi080(x1k.numpy())
    assert x800.shape == (2, 8, 800)
    print("pf800 crop OK", x800.shape)

    def _smoke_21_predictor(*, n_times: int = 1000, pf080: bool = False):
        arm = ARMS["F_mi_a"] if not pf080 else ARMS["F_mi_080"]
        m21 = MaskFutureDualExpert(
            n_times=n_times,
            n_outputs=3,
            use_predictor=True,
        )
        xf = torch.randn(2, 8, n_times)
        xm = m21.make_mask(xf)
        o21 = m21(xm, x_full=xf, train_mode=True)
        assert o21["p_cur"].shape == (2, 3)
        assert "p_final" not in o21 or o21.get("p_final") is o21["p_cur"]
        loss21, meta21 = compute_losses(
            o21,
            y,
            xf,
            lambda_cls=1,
            lambda_pred=1,
            lambda_sig=0.05,
            lambda_dec=0,
            cls_cur=True,
            cls_final=False,
            cls_future=False,
            use_sigreg=True,
            sigreg=SIGReg(),
        )
        loss21.backward()
        print(
            f"21 predictor n_times={n_times} OK",
            f"l_pred={meta21.get('l_pred', 0):.4f}",
        )

    _smoke_21_predictor(n_times=1000)
    _smoke_21_predictor(n_times=800, pf080=True)

    # J1_tok：同窗块掩码 + token L_pred
    arm_j1 = ARMS["J1_tok"]
    mj1 = MaskFutureDualExpert(
        n_times=1000,
        n_outputs=3,
        use_predictor=True,
        predictor_pos_token=arm_j1.predictor_pos_token,
        inwin_jepa=True,
        n_inwin_blocks=int(arm_j1.extra.get("n_inwin_blocks", 4)),
    )
    bs = sample_block_starts(2, device=x.device)
    oj1 = mj1(
        mj1.make_mask(x),
        x_full=x,
        train_mode=True,
        inwin_block_starts=bs,
    )
    assert oj1["z_pre_future_seq"].shape == (2, 4, 40)
    assert oj1["z_target_future_seq"].shape == (2, 4, 40)
    lj1, mj1meta = compute_losses(
        oj1,
        y,
        x,
        lambda_cls=1,
        lambda_pred=1,
        lambda_sig=0.05,
        lambda_dec=0,
        cls_cur=True,
        cls_final=False,
        cls_future=False,
        use_sigreg=True,
        sigreg=SIGReg(),
    )
    lj1.backward()
    print("J1_tok inwin OK", f"l_pred={mj1meta.get('l_pred'):.4f}")

    # J1_mlp：MLP 单向量 L_pred（附报）
    mj1m = MaskFutureDualExpert(
        n_times=1000,
        n_outputs=3,
        use_predictor=True,
        inwin_jepa=True,
        n_inwin_blocks=4,
    )
    oj1m = mj1m(
        mj1m.make_mask(x),
        x_full=x,
        train_mode=True,
        inwin_block_starts=bs,
    )
    assert oj1m["z_pre_future"].shape == (2, 40)
    assert oj1m["z_target_future"].shape == (2, 40)
    print("J1_mlp inwin OK")

    # A1_800：无 Predictor
    ma1 = MaskFutureDualExpert(n_times=800, n_outputs=3, use_predictor=False)
    xa = torch.randn(2, 8, 800)
    oa = ma1(ma1.make_mask(xa), train_mode=False)
    assert oa["p_cur"].shape == (2, 3)
    print("A1_800 OK")

    m800 = MaskFutureDualExpert(n_times=800, use_predictor=True)
    ratio800 = assert_future_perturbation(
        m800.encoder,
        i_vis=m800.i_vis,
        i_fut=m800.i_fut,
        n_times=800,
        future_pts=200,
    )
    print(f"pf800 future-perturb OK ratio={ratio800:.3f}")

    print("forward OK", {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})
    print("t_prime", m.t_prime, "i_vis", m.i_vis[0], "..", m.i_vis[-1], "i_fut", m.i_fut[0])


if __name__ == "__main__":
    main()
