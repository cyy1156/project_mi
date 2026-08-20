"""掩码未来 + 双专家门控模型（自写 Shallow encoder）。

U 系列开关（相对 P2，默认关）：
  - predictor_temporal：Z_vis 序列 → Temporal Attention + TCN → z_pre
  - use_spectral_decoder：z_pre → μ/β（无 8×400 波形重建）
  - gate_entropy：Gate([z_vis, z_pre, H(p_cur), H(p_future)])

T 系列开关（相对 P2，默认关）：
  - predictor_query：Future Query + Cross-Attn → Z_pre (B,L_fut,D)
  - pred_token_seq：L_pred 对齐 token 序列（非 mean 向量）
  - phase_conditioning：Query += E_phase(metadata 查表)
  - phase_aux：Future token phase 辅助 CE（T1-aux）
  - expert_attn_pool：Expert 用 AttentionPool 而非 mean
"""
from __future__ import annotations

import copy
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from _paths import SELF_MODEL
from feat_index import segment_indices
from phase_lookup import N_PHASE, future_phase_ids_fast

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


class TemporalPredictor(nn.Module):
    """U1：可见段时间序列 → 未来表征向量 (B, D)。

    结构：1×1 Conv → MultiheadAttention → Dilated TCN → mean pool → Linear。
    """

    def __init__(self, dim: int, dropout: float = 0.3, n_heads: int = 4):
        super().__init__()
        self.dim = int(dim)
        heads = n_heads
        while heads > 1 and self.dim % heads != 0:
            heads -= 1
        self.proj = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(
            self.dim, heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.dim)
        self.tcn = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(self.dim, self.dim)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: (B, D, T_vis)
        x = self.proj(z_seq)  # (B, D, T)
        tok = x.transpose(1, 2)  # (B, T, D)
        attn_out, _ = self.attn(tok, tok, tok, need_weights=False)
        tok = self.attn_norm(tok + attn_out)
        y = self.tcn(tok.transpose(1, 2))  # (B, D, T)
        pooled = y.mean(dim=-1)
        return self.out(pooled)


class AttentionPool(nn.Module):
    """(B, L, D) → (B, D) 可学习 query 加权池化。"""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, int(dim)))
        nn.init.normal_(self.q, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        logits = (x * self.q).sum(dim=-1)  # (B, L)
        w = torch.softmax(logits, dim=-1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class QueryFuturePredictor(nn.Module):
    """T1：Q_future = E_pos (+ E_phase) → CrossAttn(H_vis) → Z_pre (B, L_fut, D)。"""

    def __init__(
        self,
        dim: int,
        n_fut: int,
        *,
        use_phase: bool = True,
        n_phase: int = N_PHASE,
        dropout: float = 0.3,
        n_heads: int = 4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_fut = int(n_fut)
        heads = n_heads
        while heads > 1 and self.dim % heads != 0:
            heads -= 1
        self.pos_emb = nn.Embedding(self.n_fut, self.dim)
        self.phase_emb = (
            nn.Embedding(int(n_phase), self.dim) if use_phase else None
        )
        self.cross = nn.MultiheadAttention(
            self.dim, heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(self.dim)
        self.out = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        h_vis: torch.Tensor,
        phase_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # h_vis: (B, L_vis, D)
        b = h_vis.size(0)
        device = h_vis.device
        pos = torch.arange(self.n_fut, device=device).unsqueeze(0).expand(b, -1)
        q = self.pos_emb(pos)
        if self.phase_emb is not None and phase_ids is not None:
            q = q + self.phase_emb(phase_ids)
        attn_out, _ = self.cross(q, h_vis, h_vis, need_weights=False)
        tok = self.norm(attn_out + q)
        return self.out(tok)


class SpectralDecoder(nn.Module):
    """U2：z_pre → (μ, β) 对数能量（相对未来段真值约束）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, max(32, dim)),
            nn.GELU(),
            nn.Linear(max(32, dim), 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (B, 2) → [mu, beta]


def _entropy_from_prob(p: torch.Tensor) -> torch.Tensor:
    """p: (B, C) → H: (B, 1)，自然对数；归一化到约 [0,1]（除 log C）。"""
    logp = torch.log(p.clamp_min(1e-8))
    h = -(p * logp).sum(dim=-1, keepdim=True)
    return h / math.log(float(p.size(-1)))


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
        predictor_temporal: bool = False,
        use_spectral_decoder: bool = False,
        gate_entropy: bool = False,
        predictor_query: bool = False,
        pred_token_seq: bool = False,
        phase_conditioning: bool = False,
        phase_aux: bool = False,
        expert_attn_pool: bool = False,
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
        self.predictor_temporal = bool(predictor_temporal)
        self.use_spectral_decoder = bool(use_spectral_decoder)
        self.gate_entropy = bool(gate_entropy)
        self.predictor_query = bool(predictor_query)
        self.pred_token_seq = bool(pred_token_seq)
        self.phase_conditioning = bool(phase_conditioning)
        self.phase_aux = bool(phase_aux)
        self.expert_attn_pool = bool(expert_attn_pool)
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
        n_fut = len(self.i_fut) if self.i_fut else 1
        self.attn_pool_vis = AttentionPool(d) if expert_attn_pool else None
        self.attn_pool_pre = AttentionPool(d) if expert_attn_pool else None
        self.expert_cur = MLP([d, 64, n_outputs], dropout=0.0)
        if use_predictor:
            if self.predictor_query:
                self.predictor = QueryFuturePredictor(
                    d,
                    n_fut,
                    use_phase=phase_conditioning,
                    dropout=pred_dropout,
                )
            elif self.predictor_temporal:
                self.predictor = TemporalPredictor(d, dropout=pred_dropout)
            else:
                self.predictor = MLP([d, 2 * d, d], dropout=pred_dropout, end_act=False)
        else:
            self.predictor = None
        self.phase_head = (
            nn.Linear(d, N_PHASE) if (phase_aux and use_predictor) else None
        )
        self.expert_future = (
            MLP([d, 64, n_outputs], dropout=0.0) if use_expert_future else None
        )
        if use_gate:
            gate_in = 2 * d + (2 if self.gate_entropy else 0)
            self.gate = nn.Sequential(
                nn.Linear(gate_in, 64), nn.GELU(), nn.Linear(64, 1)
            )
        else:
            self.gate = None
        # 波形 Decoder（P2）与 Spectral Decoder（U2）互斥优先：spectral 开则不用波形
        if use_spectral_decoder:
            self.decoder = None
            self.spectral_decoder = SpectralDecoder(d)
        else:
            self.spectral_decoder = None
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

    def _feat_bt(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 4:
            feat = feat.squeeze(-1)
        return feat

    def _pool_segments(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # feat: (B, D, T', 1) → (B, D, T')
        feat = self._feat_bt(feat)
        z_vis = feat[:, :, self.i_vis].mean(dim=-1)
        z_fut = None
        if self.i_fut:
            z_fut = feat[:, :, self.i_fut].mean(dim=-1)
        return z_vis, z_fut

    def _vis_sequence(self, feat: torch.Tensor) -> torch.Tensor:
        feat = self._feat_bt(feat)
        return feat[:, :, self.i_vis]  # (B, D, T_vis)

    def _fut_sequence(self, feat: torch.Tensor) -> torch.Tensor | None:
        if not self.i_fut:
            return None
        feat = self._feat_bt(feat)
        return feat[:, :, self.i_fut]  # (B, D, T_fut)

    def _seq_to_tokens(self, z_seq: torch.Tensor) -> torch.Tensor:
        """(B, D, L) → (B, L, D)。"""
        return z_seq.transpose(1, 2).contiguous()

    def _read_vis(self, feat: torch.Tensor) -> torch.Tensor:
        h = self._seq_to_tokens(self._vis_sequence(feat))
        if self.attn_pool_vis is not None:
            return self.attn_pool_vis(h)
        return h.mean(dim=1)

    def _read_pre(self, z_pre_seq: torch.Tensor) -> torch.Tensor:
        if self.attn_pool_pre is not None:
            return self.attn_pool_pre(z_pre_seq)
        return z_pre_seq.mean(dim=1)

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
        t0_sec: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        no_grad_target: bool = True,
        ema_target: bool = False,
        train_mode: bool = True,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        feat_m = self.encoder.forward_features(x_mask)
        if self.predictor_query or self.expert_attn_pool:
            z_vis = self._read_vis(feat_m)
        else:
            z_vis, _ = self._pool_segments(feat_m)
        out["z_mask_vis"] = z_vis
        logits_cur = self.expert_cur(z_vis)
        out["logits_cur"] = logits_cur
        p_cur = F.softmax(logits_cur, dim=-1)
        out["p_cur"] = p_cur

        z_pre = None
        z_pre_seq = None
        phase_ids = None
        if self.predictor is not None:
            if self.predictor_query:
                h_vis = self._seq_to_tokens(self._vis_sequence(feat_m))
                if self.phase_conditioning and t0_sec is not None and self.i_fut:
                    # Phase 仅时间几何；不得传入 y（Rest 标签侧信道）
                    phase_ids = future_phase_ids_fast(
                        t0_sec.float(),
                        self.i_fut,
                        n_times=self.n_times,
                        t_prime=self.t_prime,
                    )
                    out["phase_ids"] = phase_ids
                z_pre_seq = self.predictor(h_vis, phase_ids)
                out["z_pre_future_seq"] = z_pre_seq
                z_pre = self._read_pre(z_pre_seq)
            elif self.predictor_temporal:
                z_pre = self.predictor(self._vis_sequence(feat_m))
            else:
                z_pre = self.predictor(z_vis)
            out["z_pre_future"] = z_pre
            if self.phase_head is not None and z_pre_seq is not None and phase_ids is not None:
                out["phase_logits"] = self.phase_head(z_pre_seq)

        if (
            train_mode
            and x_full is not None
            and self.predictor is not None
            and self.n_times >= 1000
        ):
            if ema_target:
                if self.ema_encoder is None:
                    raise RuntimeError("ema_target=True 但未 init_ema_encoder()")
                enc_f = self.ema_encoder
                detached = True
            elif no_grad_target:
                enc_f = self.encoder
                detached = True
            else:
                enc_f = self.encoder
                detached = False
            out["target_detached"] = detached
            if detached:
                with torch.no_grad():
                    feat_f = enc_f.forward_features(x_full)
            else:
                feat_f = enc_f.forward_features(x_full)
            if self.pred_token_seq and self.i_fut:
                z_tgt_seq = self._seq_to_tokens(self._fut_sequence(feat_f))
                out["z_target_future_seq"] = z_tgt_seq
                z_tgt = z_tgt_seq.mean(dim=1)
            else:
                _, z_tgt = self._pool_segments(feat_f)
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
            if self.gate_entropy:
                h_cur = _entropy_from_prob(p_cur)
                h_fut = _entropy_from_prob(p_future)
                g_in = torch.cat([z_vis, z_pre, h_cur, h_fut], dim=-1)
                out["H_cur"] = h_cur
                out["H_future"] = h_fut
            else:
                g_in = torch.cat([z_vis, z_pre], dim=-1)
            alpha = torch.sigmoid(self.gate(g_in))
            out["alpha"] = alpha
            out["p_final"] = alpha * p_cur + (1.0 - alpha) * p_future
        else:
            out["p_final"] = p_cur

        if self.decoder is not None and z_pre is not None:
            x_hat = self.decoder(z_pre).view(x_mask.size(0), -1, 400)
            out["x_hat_future"] = x_hat
        if self.spectral_decoder is not None and z_pre is not None:
            out["band_hat"] = self.spectral_decoder(z_pre)  # (B, 2)

        return out
