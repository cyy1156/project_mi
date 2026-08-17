"""掩码未来 + 双专家门控模型（自写 Shallow encoder）。"""
from __future__ import annotations

import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from _paths import SELF_MODEL
from feat_index import segment_indices

if str(SELF_MODEL) not in sys.path:
    sys.path.insert(0, str(SELF_MODEL))

from shallowfbcsp import ShallowFBCSPNet  # noqa: E402


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0, end_act: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            elif end_act:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskFutureDualExpert(nn.Module):
    """
    输入 X: (B, 8, T)  T=500(A0) 或 1000(pf)。
    A0 仅用 encoder+Expert_cur（等价单专家）。
    """

    def __init__(
        self,
        *,
        n_chans: int = 8,
        n_times: int = 1000,
        n_outputs: int = 3,
        embed_dim: int = 40,
        drop_prob: float = 0.5,
        pred_dropout: float = 0.3,
        use_predictor: bool = False,
        use_expert_future: bool = False,
        use_gate: bool = False,
        use_decoder: bool = False,
        mask_learnable: bool = False,
        fixed_alpha: float | None = None,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.n_times = int(n_times)
        self.embed_dim = int(embed_dim)
        self.use_predictor = use_predictor
        self.use_expert_future = use_expert_future
        self.use_gate = use_gate
        self.use_decoder = use_decoder
        self.fixed_alpha = fixed_alpha
        self.ema_momentum = float(ema_momentum)
        self.ema_encoder: nn.Module | None = None

        self.encoder = ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            n_filters_time=embed_dim,
            n_filters_spat=embed_dim,
            drop_prob=drop_prob,
        )
        # 探测 T'
        with torch.inference_mode():
            z = self.encoder.forward_features(torch.zeros(1, n_chans, n_times))
            # (B, D, T', 1) or (B, D, T')
            if z.ndim == 4:
                self.t_prime = int(z.shape[2])
            else:
                self.t_prime = int(z.shape[-1])
        if n_times == 1000:
            self.i_vis, self.i_fut = segment_indices(n_times, self.t_prime)
            # 若 T' 非 61，仍按 feat_index(600) 切，并记录
            if self.t_prime != 61:
                from feat_index import feat_index

                i_split = feat_index(600, n_times, self.t_prime)
                self.i_vis = list(range(0, i_split))
                self.i_fut = list(range(i_split, self.t_prime))
        else:
            self.i_vis, self.i_fut = list(range(self.t_prime)), []

        d = embed_dim
        self.expert_cur = MLP([d, 64, n_outputs], dropout=0.0)
        self.predictor = (
            MLP([d, 2 * d, d], dropout=pred_dropout, end_act=False)
            if use_predictor
            else None
        )
        self.expert_future = (
            MLP([d, 64, n_outputs], dropout=0.0) if use_expert_future else None
        )
        self.gate = (
            nn.Sequential(nn.Linear(2 * d, 64), nn.GELU(), nn.Linear(64, 1))
            if use_gate
            else None
        )
        self.decoder = (
            nn.Linear(d, n_chans * 400) if use_decoder else None
        )
        self.mask_token = (
            nn.Parameter(torch.zeros(1, n_chans, 400)) if mask_learnable else None
        )
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=0.02)

    def init_ema_encoder(self) -> None:
        """B10：独立 EMA encoder（不共享权重）。"""
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False
        self.ema_encoder.eval()

    @torch.no_grad()
    def update_ema_encoder(self) -> None:
        if self.ema_encoder is None:
            return
        m = self.ema_momentum
        for ps, pt in zip(self.encoder.parameters(), self.ema_encoder.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
        for bs, bt in zip(self.encoder.buffers(), self.ema_encoder.buffers()):
            bt.copy_(bs)

    def _pool_segments(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # feat: (B, D, T', 1) → (B, D, T')
        if feat.ndim == 4:
            feat = feat.squeeze(-1)
        z_vis = feat[:, :, self.i_vis].mean(dim=-1)
        z_fut = None
        if self.i_fut:
            z_fut = feat[:, :, self.i_fut].mean(dim=-1)
        return z_vis, z_fut

    def make_mask(self, x_full: torch.Tensor) -> torch.Tensor:
        x = x_full.clone()
        if self.n_times < 1000:
            return x
        if self.mask_token is not None:
            x[..., -400:] = self.mask_token.expand(x.size(0), -1, -1)
        else:
            x[..., -400:] = 0
        return x

    def forward(
        self,
        x_mask: torch.Tensor,
        x_full: torch.Tensor | None = None,
        *,
        no_grad_target: bool = True,
        ema_target: bool = False,
        train_mode: bool = True,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        feat_m = self.encoder.forward_features(x_mask)
        z_vis, _ = self._pool_segments(feat_m)
        out["z_mask_vis"] = z_vis
        logits_cur = self.expert_cur(z_vis)
        out["logits_cur"] = logits_cur
        p_cur = F.softmax(logits_cur, dim=-1)
        out["p_cur"] = p_cur

        z_pre = None
        if self.predictor is not None:
            z_pre = self.predictor(z_vis)
            out["z_pre_future"] = z_pre

        if (
            train_mode
            and x_full is not None
            and self.predictor is not None
            and self.n_times >= 1000
        ):
            if ema_target:
                if self.ema_encoder is None:
                    raise RuntimeError("ema_target=True 但未 init_ema_encoder()")
                with torch.no_grad():
                    feat_f = self.ema_encoder.forward_features(x_full)
                    _, z_tgt = self._pool_segments(feat_f)
                out["target_detached"] = True
            elif no_grad_target:
                with torch.no_grad():
                    feat_f = self.encoder.forward_features(x_full)
                    _, z_tgt = self._pool_segments(feat_f)
                out["target_detached"] = True
            else:
                # B2：target 路可回传（losses 侧不再强制 detach）
                feat_f = self.encoder.forward_features(x_full)
                _, z_tgt = self._pool_segments(feat_f)
                out["target_detached"] = False
            if z_tgt is not None:
                out["z_target_future"] = z_tgt

        p_future = None
        if self.expert_future is not None and z_pre is not None:
            logits_fut = self.expert_future(z_pre)
            out["logits_future"] = logits_fut
            p_future = F.softmax(logits_fut, dim=-1)
            out["p_future"] = p_future

        if self.fixed_alpha is not None and p_future is not None:
            a = float(self.fixed_alpha)
            alpha = x_mask.new_full((x_mask.size(0), 1), a)
            out["alpha"] = alpha
            # α=1 / α=0 复用同一对象，避免与 cls_cur 重复计 CE
            if abs(a - 1.0) < 1e-8:
                out["p_final"] = p_cur
            elif abs(a - 0.0) < 1e-8:
                out["p_final"] = p_future
            else:
                out["p_final"] = alpha * p_cur + (1.0 - alpha) * p_future
        elif self.gate is not None and z_pre is not None and p_future is not None:
            alpha = torch.sigmoid(self.gate(torch.cat([z_vis, z_pre], dim=-1)))
            out["alpha"] = alpha
            out["p_final"] = alpha * p_cur + (1.0 - alpha) * p_future
        else:
            out["p_final"] = p_cur

        if self.decoder is not None and z_pre is not None:
            x_hat = self.decoder(z_pre).view(x_mask.size(0), -1, 400)
            out["x_hat_future"] = x_hat

        return out
