"""多任务损失。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sigreg import SIGReg


def ce_from_prob(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # p: (B,C) 概率；稳定写法走 logits 更好，此处兼容已 softmax
    logp = torch.log(p.clamp_min(1e-8))
    return F.nll_loss(logp, y)


def compute_losses(
    out: dict,
    y: torch.Tensor,
    x_full: torch.Tensor | None,
    *,
    lambda_cls: float,
    lambda_pred: float,
    lambda_sig: float,
    lambda_dec: float,
    cls_cur: bool,
    cls_final: bool,
    cls_future: bool,
    use_sigreg: bool,
    sigreg: SIGReg | None,
    dec_no_psd: bool = False,
    dec_no_mubeta: bool = False,
    dec_no_time: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    parts: list[torch.Tensor] = []
    meta: dict[str, float] = {}

    l_cls = y.new_zeros(()).float()
    n_cls = 0
    if cls_cur and "logits_cur" in out:
        l_cls = l_cls + F.cross_entropy(out["logits_cur"], y)
        n_cls += 1
    if cls_final and "p_final" in out:
        # 有 gate 时 p_final 为概率混合；用 NLL
        if not (cls_cur and out["p_final"] is out.get("p_cur")):
            l_cls = l_cls + ce_from_prob(out["p_final"], y)
            n_cls += 1
    if cls_future and "logits_future" in out:
        l_cls = l_cls + F.cross_entropy(out["logits_future"], y)
        n_cls += 1
    if n_cls > 0 and lambda_cls != 0:
        parts.append(lambda_cls * l_cls)
        meta["l_cls"] = float(l_cls.detach())

    if (
        lambda_pred > 0
        and "z_pre_future" in out
        and "z_target_future" in out
    ):
        z_tgt = out["z_target_future"]
        # 默认 / EMA / no_grad：detach；B2（target_detached=False）允许回传
        if out.get("target_detached", True):
            z_tgt = z_tgt.detach()
        l_pred = F.mse_loss(out["z_pre_future"], z_tgt)
        parts.append(lambda_pred * l_pred)
        meta["l_pred"] = float(l_pred.detach())

    if use_sigreg and lambda_sig > 0 and sigreg is not None and "z_mask_vis" in out:
        l_sig = sigreg(out["z_mask_vis"])
        parts.append(lambda_sig * l_sig)
        meta["l_sig"] = float(l_sig.detach())

    if (
        lambda_dec > 0
        and "band_hat" in out
        and x_full is not None
        and x_full.size(-1) >= 1000
    ):
        # U2：直接约束 μ/β（对未来 400pt 通道均值波形估真值）
        x_fut = x_full[..., -400:]
        fut_m = x_fut.mean(dim=1).float()
        freqs = torch.fft.rfftfreq(400, d=1.0 / 250.0).to(fut_m.device)
        psd = torch.fft.rfft(fut_m, dim=-1).abs()

        def _band(lo: float, hi: float) -> torch.Tensor:
            m = (freqs >= lo) & (freqs < hi)
            # (B,)
            return psd[..., m].mean(dim=-1).clamp_min(1e-8).log()

        tgt = torch.stack([_band(8.0, 13.0), _band(13.0, 30.0)], dim=-1)  # (B,2)
        l_spec = F.mse_loss(out["band_hat"].float(), tgt.detach())
        parts.append(lambda_dec * l_spec)
        meta["l_dec"] = float(l_spec.detach())
        meta["l_spec"] = float(l_spec.detach())
    elif (
        lambda_dec > 0
        and "x_hat_future" in out
        and x_full is not None
        and x_full.size(-1) >= 1000
    ):
        x_fut = x_full[..., -400:]
        x_hat = out["x_hat_future"]
        l_dec = x_fut.new_zeros(())
        # 通道均值后再估能量（开跑）
        hat_m = x_hat.mean(dim=1)
        fut_m = x_fut.mean(dim=1)
        if not dec_no_time:
            l_dec = l_dec + 0.1 * F.mse_loss(hat_m, fut_m)
        if not dec_no_psd:
            # 简易 rfft PSD L1（AMP/half 下 cuFFT 要求长度 2^n，故强制 float32）
            hat32 = hat_m.float()
            fut32 = fut_m.float()
            psd_h = torch.fft.rfft(hat32, dim=-1).abs()
            psd_f = torch.fft.rfft(fut32, dim=-1).abs()
            l_dec = l_dec + (psd_h - psd_f).abs().mean().to(dtype=l_dec.dtype)
        if not dec_no_mubeta:
            # 粗频带能量：用 rfft bin 近似 μ/β（fs=250）
            # μ 8–13, β 13–30 → 略
            freqs = torch.fft.rfftfreq(400, d=1.0 / 250.0).to(hat_m.device)

            def band_e(x, lo, hi):
                p = torch.fft.rfft(x.float(), dim=-1).abs()
                m = (freqs >= lo) & (freqs < hi)
                return p[..., m].mean()

            l_dec = l_dec + (band_e(hat_m, 8, 13) - band_e(fut_m, 8, 13)).abs().to(
                dtype=l_dec.dtype
            )
            l_dec = l_dec + (band_e(hat_m, 13, 30) - band_e(fut_m, 13, 30)).abs().to(
                dtype=l_dec.dtype
            )
        parts.append(lambda_dec * l_dec)
        meta["l_dec"] = float(l_dec.detach())

    if not parts:
        total = y.new_zeros(()).float()
    else:
        total = parts[0]
        for p in parts[1:]:
            total = total + p
    meta["l_total"] = float(total.detach())
    return total, meta
