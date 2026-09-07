"""方案 19 · DualBandShallow：μ/β 各 TemporalConv(20)+Spatial(20) + Gate。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SELF_MODEL = REPO / "self_model"
if str(SELF_MODEL) not in sys.path:
    sys.path.insert(0, str(SELF_MODEL))

from shallowfbcsp import ShallowFBCSPNet  # noqa: E402  # pyright: ignore[reportMissingImports]


def _branch_logits_and_z(
    branch: ShallowFBCSPNet, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B,8,T) → logits (B,C), z (B,F) mean-pool features."""
    feat = branch.forward_features(x)
    z = feat.mean(dim=(2, 3))
    logits = branch.final_layer(feat)
    if logits.ndim > 2:
        logits = logits.reshape(logits.shape[0], -1)
    return logits, z


class DualBandShallowGate(nn.Module):
    """
    V2：两路独立 Shallow（各 n_filters=20）+ 融合。
    fuse:
      - gate：α=σ(Gate([z_μ,z_β]))，p_final=α p_μ+(1-α)p_β
      - fixed05：α=0.5
      - concat：Linear(concat(z_μ,z_β)) → logits_final（仍可算分支 CE）
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        *,
        drop_prob: float = 0.5,
        n_filters_branch: int = 20,
        fuse: str = "gate",
        lambda_aux: float = 0.5,
    ) -> None:
        super().__init__()
        self.fuse = str(fuse).lower()
        self.lambda_aux = float(lambda_aux)
        nf = int(n_filters_branch)
        kw = dict(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            n_filters_time=nf,
            n_filters_spat=nf,
            drop_prob=drop_prob,
            attn=None,
        )
        self.branch_mu = ShallowFBCSPNet(**kw)
        self.branch_beta = ShallowFBCSPNet(**kw)
        if self.fuse == "gate":
            self.gate = nn.Sequential(
                nn.Linear(nf * 2, max(16, nf)),
                nn.GELU(),
                nn.Linear(max(16, nf), 1),
            )
            self.concat_head = None
        elif self.fuse == "fixed05":
            self.gate = None
            self.concat_head = None
        elif self.fuse == "concat":
            self.gate = None
            self.concat_head = nn.Linear(nf * 2, n_outputs)
        else:
            raise ValueError(f"unknown fuse={fuse!r}; use gate|fixed05|concat")

    def forward(
        self,
        x_mu: torch.Tensor,
        x_beta: torch.Tensor,
        *,
        return_parts: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        logits_mu, z_mu = _branch_logits_and_z(self.branch_mu, x_mu)
        logits_beta, z_beta = _branch_logits_and_z(self.branch_beta, x_beta)
        p_mu = F.softmax(logits_mu, dim=-1)
        p_beta = F.softmax(logits_beta, dim=-1)

        if self.fuse == "concat":
            assert self.concat_head is not None
            logits_final = self.concat_head(torch.cat([z_mu, z_beta], dim=-1))
            p_final = F.softmax(logits_final, dim=-1)
            alpha = None
        else:
            if self.fuse == "gate":
                assert self.gate is not None
                alpha = torch.sigmoid(self.gate(torch.cat([z_mu, z_beta], dim=-1)))
            else:
                alpha = x_mu.new_full((x_mu.size(0), 1), 0.5)
            p_final = alpha * p_mu + (1.0 - alpha) * p_beta
            logits_final = torch.log(p_final.clamp_min(1e-8))

        if return_parts:
            out = {
                "logits_final": logits_final,
                "logits_mu": logits_mu,
                "logits_beta": logits_beta,
                "p_final": p_final,
                "p_mu": p_mu,
                "p_beta": p_beta,
                "z_mu": z_mu,
                "z_beta": z_beta,
            }
            if alpha is not None:
                out["alpha"] = alpha
            return out
        return logits_final

    def loss(
        self,
        parts: dict[str, torch.Tensor],
        y: torch.Tensor,
        *,
        criterion: nn.Module | None = None,
    ) -> torch.Tensor:
        ce = criterion or nn.CrossEntropyLoss()
        # concat：logits_final 为真实 logit；gate/fixed：为 log p
        if self.fuse == "concat":
            l_main = ce(parts["logits_final"], y)
        else:
            l_main = F.nll_loss(parts["logits_final"], y)
        if self.lambda_aux > 0:
            l_aux = 0.5 * (ce(parts["logits_mu"], y) + ce(parts["logits_beta"], y))
            return l_main + self.lambda_aux * l_aux
        return l_main


def count_params(m: nn.Module) -> int:
    return sum(int(p.numel()) for p in m.parameters() if p.requires_grad)
