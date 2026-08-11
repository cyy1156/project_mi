"""ShallowFBCSPNet 结构增强变体定义（S1~S5）。

严格按方案 `资料/模型训练/09_旁路_shallow_网络结构增强_Three_openbmi_accpaper/方案.md` 实现。

S0 定义在 baseline_shallow_s0.py，本文件仅含 S1~S5。

每个变体返回一个 build_model 函数，签名：
    build_model(n_chans, n_times, n_outputs, drop_prob) -> nn.Module
模型 forward 接收 (B, n_chans, n_times)，输出 (B, n_outputs)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import ShallowFBCSPNet


# ══════════════════════════════════════════════════════════════════
# S1：结构超参网格（顺序搜索 S1a→S1b→S1c→S1d）
# ══════════════════════════════════════════════════════════════════
# S0 默认值：n_filters_time=40, filter_time_length=25, n_filters_spat=40,
#            pool_time_length=75, pool_time_stride=15, drop_prob=0.5
#
# 方案要求：
#   S1a: filter_time_length ∈ {13, 25, 50}       — 其余=S0默认
#   S1b: n_filters ∈ {20, 40, 64} (time=spat)    — 在S1a最优核长上
#   S1c: pool_time_stride ∈ {10, 15, 25}          — 在S1b最优上
#   S1d: drop_prob ∈ {0.25, 0.5}                  — 在S1c最优上
# pool_time_length 不是搜索因子，固定 75。


@dataclass
class S1Base:
    """从前序阶段最优继承的基础参数（默认=S0）。"""
    filter_time_length: int = 25      # S1a 最优
    n_filters: int = 40               # S1b 最优 (time=spat)
    pool_time_stride: int = 15        # S1c 最优


# 全局可配（run_arm.py 通过命令行设置）
S1_BASE = S1Base()


# ── S1a：扫 filter_time_length ─────────────────────────────────
S1A_CANDIDATES = [13, 25, 50]


def build_s1a(t_length: int):
    """S1a: 只改 filter_time_length，其余=S0默认。"""
    def _build(n_chans, n_times, n_outputs, drop_prob):
        return ShallowFBCSPNet(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
            n_filters_time=40, filter_time_length=t_length,
            n_filters_spat=40, pool_time_length=75, pool_time_stride=15,
            drop_prob=drop_prob,
        )
    return _build


# ── S1b：扫 n_filters (time=spat) ──────────────────────────────
S1B_CANDIDATES = [20, 40, 64]


def build_s1b(n_filters: int):
    """S1b: 在 S1a 最优核长上扫 n_filters (time=spat)。"""
    def _build(n_chans, n_times, n_outputs, drop_prob):
        return ShallowFBCSPNet(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
            n_filters_time=n_filters, filter_time_length=S1_BASE.filter_time_length,
            n_filters_spat=n_filters, pool_time_length=75, pool_time_stride=15,
            drop_prob=drop_prob,
        )
    return _build


# ── S1c：扫 pool_time_stride ───────────────────────────────────
S1C_CANDIDATES = [10, 15, 25]


def build_s1c(pool_stride: int):
    """S1c: 在 S1b 最优上扫 pool_time_stride。"""
    def _build(n_chans, n_times, n_outputs, drop_prob):
        return ShallowFBCSPNet(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
            n_filters_time=S1_BASE.n_filters, filter_time_length=S1_BASE.filter_time_length,
            n_filters_spat=S1_BASE.n_filters, pool_time_length=75, pool_time_stride=pool_stride,
            drop_prob=drop_prob,
        )
    return _build


# ── S1d：扫 drop_prob ──────────────────────────────────────────
S1D_CANDIDATES = [0.25, 0.5]


def build_s1d(drop_override: float):
    """S1d: 在 S1c 最优上扫 drop_prob（覆盖传入的 drop_prob）。"""
    def _build(n_chans, n_times, n_outputs, drop_prob):
        return ShallowFBCSPNet(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
            n_filters_time=S1_BASE.n_filters, filter_time_length=S1_BASE.filter_time_length,
            n_filters_spat=S1_BASE.n_filters, pool_time_length=75,
            pool_time_stride=S1_BASE.pool_time_stride,
            drop_prob=drop_override,
        )
    return _build


def get_s1_arm(arm: str) -> tuple[Callable, str, dict]:
    """解析 S1 臂名，返回 (build_fn, note, meta)。"""
    if arm.startswith("s1a_t"):
        t = int(arm.split("_t")[1])
        return build_s1a(t), f"S1a filter_time_length={t}", {
            "shallow": {"variant": "S1a", "filter_time_length": t}, "accpaper": True}
    if arm.startswith("s1b_f"):
        f = int(arm.split("_f")[1])
        return build_s1b(f), f"S1b n_filters={f} (time=spat), t_len={S1_BASE.filter_time_length}", {
            "shallow": {"variant": "S1b", "n_filters": f, "base_t": S1_BASE.filter_time_length},
            "accpaper": True}
    if arm.startswith("s1c_ps"):
        ps = int(arm.split("_ps")[1])
        return build_s1c(ps), f"S1c pool_stride={ps}, f={S1_BASE.n_filters}, t_len={S1_BASE.filter_time_length}", {
            "shallow": {"variant": "S1c", "pool_stride": ps, "base_f": S1_BASE.n_filters,
                        "base_t": S1_BASE.filter_time_length}, "accpaper": True}
    if arm.startswith("s1d_d"):
        d = int(arm.split("_d")[1]) / 100  # d025→0.25, d050→0.5
        return build_s1d(d), f"S1d drop={d}, ps={S1_BASE.pool_time_stride}, f={S1_BASE.n_filters}", {
            "shallow": {"variant": "S1d", "drop_prob": d, "base_ps": S1_BASE.pool_time_stride,
                        "base_f": S1_BASE.n_filters}, "accpaper": True}
    raise ValueError(f"未知 S1 臂: {arm}")


# ══════════════════════════════════════════════════════════════════
# S2：多尺度时间核（中风险）
# ══════════════════════════════════════════════════════════════════
# 方案要求：
#   S2_ms_concat: 三分支 TimeConv(13/25/50) 通道拼接 → 共享 SpatConv→Square→Pool→Log→头
#   S2_ms_sum:    三分支各自到 log-power 后相加 → 分类
# 约束：总滤波器预算 ~40（如 16+16+8）；保留 Square+mean pool+SafeLog


class _MultiScaleConcat(nn.Module):
    """S2_ms_concat: 三分支时间核 → 通道拼接 → 共享 SpatConv→Square→Pool→Log→头。"""

    def __init__(self, n_chans, n_times, n_outputs, drop_prob,
                 kernel_lengths=(13, 25, 50), filter_budget=(16, 16, 8)):
        super().__init__()
        assert len(kernel_lengths) == len(filter_budget)
        total_filters = sum(filter_budget)
        # 三分支 TimeConv（same padding 保持 T 一致）
        self.time_convs = nn.ModuleList([
            nn.Conv2d(1, f, (k, 1), padding=(k // 2, 0))
            for k, f in zip(kernel_lengths, filter_budget)
        ])
        # 共享 SpatConv
        self.spat_conv = nn.Conv2d(total_filters, total_filters, (1, n_chans))
        self.bn = nn.BatchNorm2d(total_filters)
        # Square + Pool + SafeLog
        self.pool_length = 75
        self.pool_stride = 15
        self.pool = nn.AvgPool2d((self.pool_length, 1), stride=(self.pool_stride, 1))
        self.drop = nn.Dropout(drop_prob)
        # 计算 T_pool（same padding 后 T'=n_times）
        t_after_conv = n_times  # same padding
        t_pool = (t_after_conv - self.pool_length) // self.pool_stride + 1
        self.final_layer = nn.Conv2d(total_filters, n_outputs, (t_pool, 1))

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B,1,C,T)
        branches = [conv(x) for conv in self.time_convs]
        x = torch.cat(branches, dim=1)      # (B, total, C, T)
        x = self.spat_conv(x)                # (B, total, 1, T)
        x = self.bn(x)
        x = x ** 2                           # Square
        x = self.pool(x)                     # (B, total, 1, T_pool)
        x = torch.log(x + 1e-6)             # SafeLog
        x = self.drop(x)
        x = self.final_layer(x)             # (B, n_out, 1, 1)
        return x.squeeze(-1).squeeze(-1)    # (B, n_out)


class _MultiScaleSum(nn.Module):
    """S2_ms_sum: 三分支各自 TimeConv→SpatConv→Square→Pool→Log，log-power 相加 → 分类。"""

    def __init__(self, n_chans, n_times, n_outputs, drop_prob,
                 kernel_lengths=(13, 25, 50), n_filters_per_branch=13):
        super().__init__()
        # 每分支独立 TimeConv + SpatConv
        self.branches = nn.ModuleList()
        for k in kernel_lengths:
            branch = nn.Sequential(
                nn.Conv2d(1, n_filters_per_branch, (k, 1), padding=(k // 2, 0)),
                nn.Conv2d(n_filters_per_branch, n_filters_per_branch, (1, n_chans)),
                nn.BatchNorm2d(n_filters_per_branch),
            )
            self.branches.append(branch)
        self.pool = nn.AvgPool2d((75, 1), stride=(15, 1))
        self.drop = nn.Dropout(drop_prob)
        t_pool = (n_times - 75) // 15 + 1
        self.final_layer = nn.Conv2d(n_filters_per_branch, n_outputs, (t_pool, 1))

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        log_powers = []
        for branch in self.branches:
            h = branch(x)           # (B, f, 1, T)
            h = h ** 2              # Square
            h = self.pool(h)        # (B, f, 1, T_pool)
            h = torch.log(h + 1e-6) # SafeLog
            log_powers.append(h)
        x = sum(log_powers)         # 逐元素相加
        x = self.drop(x)
        x = self.final_layer(x)
        return x.squeeze(-1).squeeze(-1)


def build_s2_ms_concat(n_chans, n_times, n_outputs, drop_prob):
    return _MultiScaleConcat(n_chans, n_times, n_outputs, drop_prob,
                             kernel_lengths=(13, 25, 50), filter_budget=(16, 16, 8))


def build_s2_ms_sum(n_chans, n_times, n_outputs, drop_prob):
    return _MultiScaleSum(n_chans, n_times, n_outputs, drop_prob,
                          kernel_lengths=(13, 25, 50), n_filters_per_branch=13)


# ══════════════════════════════════════════════════════════════════
# S3：读出头增强（中低风险 · Three 专攻）
# ══════════════════════════════════════════════════════════════════
# 方案要求：
#   S3_mlp:  AdaptiveAvgPool/Flatten → MLP(n_filters→64→n_out) + Dropout
#   S3_stats: 对 T 维做 mean/std/max 拼接 → 线性或浅 MLP
#   S3_hier:  共享骨干；Task 二类头 + left/right 头；推理先 Task 再左右
#   S3_three_only_tune: 冻结 S0/S1 卷积，只训新 Three 头
# 在 S0 或 S1 最优骨干上替换 final_layer。


def _make_shallow_backbone(n_chans, n_times, drop_prob, **kwargs):
    """创建 ShallowFBCSPNet，替换 final_layer 为 Identity，返回 (backbone, n_features, t_pool)。"""
    model = ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=1, n_times=n_times,
        drop_prob=drop_prob, **kwargs,
    )
    model.final_layer = nn.Identity()
    with torch.no_grad():
        dummy = torch.zeros(1, n_chans, n_times)
        feat = model(dummy)
    # feat: (1, C, 1, T_pool) or (1, C, T_pool)
    if feat.ndim == 4:
        n_feat = int(feat.shape[1])
        t_pool = int(feat.shape[-1])
    elif feat.ndim == 3:
        n_feat = int(feat.shape[1])
        t_pool = int(feat.shape[-1])
    else:
        n_feat = int(feat.shape[0])
        t_pool = 1
    return model, n_feat, t_pool


class _MLPHead(nn.Module):
    """S3_mlp: AdaptiveAvgPool → Flatten → MLP(n_feat→64→n_out) + Dropout。"""

    def __init__(self, backbone, n_feat, n_outputs, drop_prob):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_feat, 64),
            nn.Dropout(drop_prob),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim == 4:
            feat = self.pool(feat)  # (B, C, 1, 1)
        elif feat.ndim == 3:
            feat = feat.unsqueeze(2)  # (B, C, 1, T)
            feat = self.pool(feat)
        return self.head(feat)


class _StatsHead(nn.Module):
    """S3_stats: 对 T 维做 mean/std/max 拼接 → 浅 MLP。"""

    def __init__(self, backbone, n_feat, n_outputs, drop_prob):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(n_feat * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        feat = self.backbone(x)
        # feat: (B, C, 1, T_pool) or (B, C, T_pool)
        if feat.ndim == 4:
            feat = feat.squeeze(2)  # (B, C, T_pool)
        mean = feat.mean(dim=-1)
        std = feat.std(dim=-1)
        mx = feat.max(dim=-1).values
        stats = torch.cat([mean, std, mx], dim=-1)  # (B, 3*C)
        return self.head(stats)


class _HierHead(nn.Module):
    """S3_hier: 共享骨干；Task 二类头 + left/right 二类头。

    forward 返回 (task_logits, lr_logits)。
    推理: 先 Task；若 Task=MI，再用 lr 头决定 left/right。
    训练: 多任务损失（Task CE + L/R CE on MI trials）。
    ⚠️ 需要 task_runner.py 特殊处理多任务训练和分层推理。
    """

    def __init__(self, backbone, n_feat, n_outputs, drop_prob):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Task 头: idle vs MI (2 类)
        self.task_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_feat, 64),
            nn.Dropout(drop_prob),
            nn.Linear(64, 2),
        )
        # L/R 头: left vs right (2 类)
        self.lr_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_feat, 64),
            nn.Dropout(drop_prob),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim == 4:
            feat = self.pool(feat)
        elif feat.ndim == 3:
            feat = feat.unsqueeze(2)
            feat = self.pool(feat)
        task_logits = self.task_head(feat)
        lr_logits = self.lr_head(feat)
        return task_logits, lr_logits


def build_s3_mlp(n_chans, n_times, n_outputs, drop_prob):
    backbone, n_feat, _ = _make_shallow_backbone(n_chans, n_times, drop_prob)
    return _MLPHead(backbone, n_feat, n_outputs, drop_prob)


def build_s3_stats(n_chans, n_times, n_outputs, drop_prob):
    backbone, n_feat, _ = _make_shallow_backbone(n_chans, n_times, drop_prob)
    return _StatsHead(backbone, n_feat, n_outputs, drop_prob)


def build_s3_hier(n_chans, n_times, n_outputs, drop_prob):
    """S3_hier: ⚠️ 返回 (task_logits, lr_logits) 元组，需 task_runner 特殊处理。"""
    backbone, n_feat, _ = _make_shallow_backbone(n_chans, n_times, drop_prob)
    return _HierHead(backbone, n_feat, n_outputs, drop_prob)


def build_s3_three_only_tune(n_chans, n_times, n_outputs, drop_prob):
    """S3_three_only_tune: 冻结骨干，只训 MLP Three 头。"""
    backbone, n_feat, _ = _make_shallow_backbone(n_chans, n_times, drop_prob)
    model = _MLPHead(backbone, n_feat, n_outputs, drop_prob)
    # 冻结骨干
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model


# ══════════════════════════════════════════════════════════════════
# S4：训练目标与试次聚合对齐（中风险）
# ══════════════════════════════════════════════════════════════════
# 方案要求：
#   S4_softvote_loss: 同 trial 窗 logits 均值后再 CE
#   S4_focal:         Focal loss (γ 扫 1-2)
#   S4_class_weight:  逆频类权
#   S4_conf_agg:      评测用置信度加权众数（仅评测，不改训练）
#
# ⚠️ S4 模型结构与 S0 相同；差异在 loss/评测，需在 task_runner.py 实现。
# S4 的 loss 函数定义在此处供 task_runner 导入。


class FocalLoss(nn.Module):
    """S4_focal: Focal loss = -α(1-p)^γ * log(p)。"""

    def __init__(self, gamma: float = 1.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        p = torch.exp(-ce)
        loss = ((1 - p) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha * loss
        return loss.mean()


def build_s4(n_chans, n_times, n_outputs, drop_prob):
    """S4 模型结构同 S0；loss/评测差异由 task_runner 处理。"""
    return ShallowFBCSPNet(
        n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
        drop_prob=drop_prob,
    )


# ══════════════════════════════════════════════════════════════════
# S5：轻量混合骨干（较高风险 · 最后）
# ══════════════════════════════════════════════════════════════════
# 方案要求：
#   S5_res_pre: Square 前加 1 个残差时序块（深度可分离/小组 Conv）
#   S5_dual:    主路 Shallow-log；旁路不加 Square 的浅 Conv；特征拼接后分类
#   S5_se:      SpatConv 后通道 SE/注意力（参数少）
# 约束：参数量 ≤ 正式 ×3（约 <5e4）；Three 无弱成功则立即停。


class _ResPreShallow(nn.Module):
    """S5_res_pre: SpatConv+BN 后、Square 前加残差时序块。"""

    def __init__(self, n_chans, n_times, n_outputs, drop_prob, **kwargs):
        super().__init__()
        backbone = ShallowFBCSPNet(
            n_chans=n_chans, n_outputs=1, n_times=n_times,
            drop_prob=drop_prob, **kwargs,
        )
        n_filt = backbone.conv_spat.out_channels
        # 残差块：深度可分离 Conv + BN
        self.res_block = nn.Sequential(
            nn.Conv2d(n_filt, n_filt, (3, 1), padding=(1, 0), groups=n_filt),  # depth-wise
            nn.Conv2d(n_filt, n_filt, (1, 1)),                                  # point-wise
            nn.BatchNorm2d(n_filt),
        )
        # 拆解 backbone 的 forward，在 Square 前插入残差块
        self.ensuredims = backbone.ensuredims
        self.dimshuffle = backbone.dimshuffle
        self.conv_time = backbone.conv_time
        self.conv_spat = backbone.conv_spat
        self.bnorm = backbone.bnorm
        self.square = backbone.square
        self.pool = backbone.pool
        self.safe_log = backbone.safe_log
        self.drop = backbone.drop
        # 计算 final_conv_length
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            h = self.ensuredims(dummy)
            h = self.dimshuffle(h)
            h = self.conv_time(h)
            h = self.conv_spat(h)
            h = self.bnorm(h)
            h = self.res_block(h) + h  # 残差
            h = self.square(h)
            h = self.pool(h)
            h = self.safe_log(h)
            t_pool = int(h.shape[-1])
        self.final_layer = nn.Conv2d(n_filt, n_outputs, (t_pool, 1))

    def forward(self, x):
        h = self.ensuredims(x)
        h = self.dimshuffle(h)
        h = self.conv_time(h)
        h = self.conv_spat(h)
        h = self.bnorm(h)
        h = self.res_block(h) + h   # 残差连接
        h = self.square(h)
        h = self.pool(h)
        h = self.safe_log(h)
        h = self.drop(h)
        h = self.final_layer(h)
        return h.squeeze(-1).squeeze(-1)


class _DualPathShallow(nn.Module):
    """S5_dual: 主路 Shallow-log + 旁路浅 Conv（无 Square），特征拼接后分类。"""

    def __init__(self, n_chans, n_times, n_outputs, drop_prob, **kwargs):
        super().__init__()
        # 主路：标准 Shallow 到 log-power
        main = ShallowFBCSPNet(
            n_chans=n_chans, n_outputs=1, n_times=n_times,
            drop_prob=drop_prob, **kwargs,
        )
        main.final_layer = nn.Identity()
        self.main = main
        n_main = main.conv_spat.out_channels

        # 旁路：浅 Conv（无 Square/Log），BN+ReLU+Pool
        n_side = 16
        self.side_conv = nn.Conv2d(1, n_side, (25, 1), padding=(12, 0))
        self.side_bn = nn.BatchNorm2d(n_side)
        self.side_pool = nn.AvgPool2d((75, 1), stride=(15, 1))

        # 干跑获取特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            main_feat = self.main(dummy)
            if main_feat.ndim == 4:
                main_feat_flat = main_feat.shape[1] * main_feat.shape[-1]
            else:
                main_feat_flat = main_feat.numel()
            side_x = dummy.unsqueeze(1)
            side_h = self.side_conv(side_x)
            side_h = self.side_bn(side_h)
            side_h = F.relu(side_h, inplace=True)
            side_h = self.side_pool(side_h)
            side_feat_flat = int(side_h.numel())

        total_feat = main_feat_flat + side_feat_flat
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_feat, 64),
            nn.Dropout(drop_prob),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        if x.ndim == 3:
            x4d = x.unsqueeze(1)
        else:
            x4d = x
        main_feat = self.main(x)     # (B, C, 1, T_pool)
        if main_feat.ndim == 4:
            main_feat = main_feat.squeeze(2)
        side_h = self.side_conv(x4d)
        side_h = self.side_bn(side_h)
        side_h = F.relu(side_h, inplace=True)
        side_h = self.side_pool(side_h)
        side_h = side_h.squeeze(2)   # (B, n_side, T_pool)
        # 对齐 T 维度（截断到较短者）
        t_min = min(main_feat.shape[-1], side_h.shape[-1])
        main_feat = main_feat[..., :t_min]
        side_h = side_h[..., :t_min]
        feat = torch.cat([main_feat, side_h], dim=1)  # (B, C_total, T_pool)
        return self.head(feat)


class _SEShallow(nn.Module):
    """S5_se: SpatConv+BN 后加 SE 通道注意力，再 Square→Pool→Log→头。"""

    def __init__(self, n_chans, n_times, n_outputs, drop_prob, reduction=4, **kwargs):
        super().__init__()
        backbone = ShallowFBCSPNet(
            n_chans=n_chans, n_outputs=1, n_times=n_times,
            drop_prob=drop_prob, **kwargs,
        )
        n_filt = backbone.conv_spat.out_channels
        # SE 块
        self.se_squeeze = nn.AdaptiveAvgPool2d(1)
        self.se_excite = nn.Sequential(
            nn.Linear(n_filt, max(n_filt // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(n_filt // reduction, 4), n_filt),
            nn.Sigmoid(),
        )
        # 拆解 backbone
        self.ensuredims = backbone.ensuredims
        self.dimshuffle = backbone.dimshuffle
        self.conv_time = backbone.conv_time
        self.conv_spat = backbone.conv_spat
        self.bnorm = backbone.bnorm
        self.n_filt = n_filt
        self.square = backbone.square
        self.pool = backbone.pool
        self.safe_log = backbone.safe_log
        self.drop = backbone.drop
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            h = self.ensuredims(dummy)
            h = self.dimshuffle(h)
            h = self.conv_time(h)
            h = self.conv_spat(h)
            h = self.bnorm(h)
            # SE
            b, c, _, _ = h.shape
            s = self.se_squeeze(h).view(b, c)
            s = self.se_excite(s).view(b, c, 1, 1)
            h = h * s
            h = self.square(h)
            h = self.pool(h)
            h = self.safe_log(h)
            t_pool = int(h.shape[-1])
        self.final_layer = nn.Conv2d(n_filt, n_outputs, (t_pool, 1))

    def forward(self, x):
        h = self.ensuredims(x)
        h = self.dimshuffle(h)
        h = self.conv_time(h)
        h = self.conv_spat(h)
        h = self.bnorm(h)
        # SE 注意力
        b, c, _, _ = h.shape
        s = self.se_squeeze(h).view(b, c)
        s = self.se_excite(s).view(b, c, 1, 1)
        h = h * s
        h = self.square(h)
        h = self.pool(h)
        h = self.safe_log(h)
        h = self.drop(h)
        h = self.final_layer(h)
        return h.squeeze(-1).squeeze(-1)


def build_s5_res_pre(n_chans, n_times, n_outputs, drop_prob):
    return _ResPreShallow(n_chans, n_times, n_outputs, drop_prob)


def build_s5_dual(n_chans, n_times, n_outputs, drop_prob):
    return _DualPathShallow(n_chans, n_times, n_outputs, drop_prob)


def build_s5_se(n_chans, n_times, n_outputs, drop_prob):
    return _SEShallow(n_chans, n_times, n_outputs, drop_prob)


# ══════════════════════════════════════════════════════════════════
# 注册表
# ══════════════════════════════════════════════════════════════════

VARIANT_REGISTRY: dict[str, dict] = {
    # S2: 多尺度时间核
    "s2_ms_concat": {"build": build_s2_ms_concat, "note": "三分支时间核(13/25/50)通道拼接，共享SpatConv"},
    "s2_ms_sum":    {"build": build_s2_ms_sum,    "note": "三分支各自log-power后相加"},
    # S3: 读出头增强
    "s3_mlp":              {"build": build_s3_mlp,              "note": "AdaptiveAvgPool→MLP(n→64→out)+Dropout"},
    "s3_stats":            {"build": build_s3_stats,            "note": "mean/std/max拼接→浅MLP"},
    "s3_hier":             {"build": build_s3_hier,             "note": "Task头+L/R头分层推理（需task_runner特殊处理）"},
    "s3_three_only_tune":  {"build": build_s3_three_only_tune,  "note": "冻结骨干，只训MLP Three头"},
    # S4: 训练目标（模型同S0，loss/评测在task_runner侧）
    "s4_softvote_loss": {"build": build_s4, "note": "trial窗logits均值后CE（需task_runner处理）"},
    "s4_focal":         {"build": build_s4, "note": "Focal loss γ扫1-2（需task_runner处理）"},
    "s4_class_weight":  {"build": build_s4, "note": "逆频类权CE（需task_runner处理）"},
    "s4_conf_agg":      {"build": build_s4, "note": "评测用置信度加权众数（仅评测不改训练）"},
    # S5: 轻量混合骨干
    "s5_res_pre": {"build": build_s5_res_pre, "note": "Square前加残差时序块"},
    "s5_dual":    {"build": build_s5_dual,    "note": "主路Shallow-log+旁路浅Conv拼接"},
    "s5_se":      {"build": build_s5_se,      "note": "SpatConv后SE通道注意力"},
}


def get_variant(arm: str) -> tuple[Callable, str, dict]:
    """返回 (build_fn, structure_note, extra_meta)。

    S0 需从 baseline_shallow_s0.py 导入。
    S1 臂名格式: s1a_t13, s1b_f20, s1c_ps10, s1d_d025
    S2/S3/S4/S5 臂名见 VARIANT_REGISTRY。
    """
    arm = arm.lower().strip()

    # S1 顺序搜索
    if arm.startswith("s1"):
        return get_s1_arm(arm)

    if arm in VARIANT_REGISTRY:
        info = VARIANT_REGISTRY[arm]
        return info["build"], f"ShallowFBCSPNet（{info['note']}）", {
            "shallow": {"backbone": "ShallowFBCSPNet", "variant": arm},
            "accpaper": True,
        }
    raise ValueError(
        f"未知 arm: {arm}; 可选: s1a_t13/s1a_t25/s1a_t50, s1b_f20/s1b_f40/s1b_f64, "
        f"s1c_ps10/s1c_ps15/s1c_ps25, s1d_d025/s1d_d050, "
        f"{', '.join(VARIANT_REGISTRY.keys())}"
    )
