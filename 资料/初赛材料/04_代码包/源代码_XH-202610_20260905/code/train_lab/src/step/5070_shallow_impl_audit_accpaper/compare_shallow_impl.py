"""方案 18 · 阶段 A/B：手写 vs braindecode ShallowFBCSP 结构与前向对齐审计。

Usage:
  python compare_shallow_impl.py
  python compare_shallow_impl.py --out D:/MI/code/train_lab/out/5070_shallow_impl_audit_accpaper/_compare
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SELF_MODEL = REPO / "self_model"
TRAIN_LAB = REPO / "code" / "train_lab"
DEFAULT_OUT = TRAIN_LAB / "out" / "5070_shallow_impl_audit_accpaper" / "_compare"

if str(SELF_MODEL) not in sys.path:
    sys.path.insert(0, str(SELF_MODEL))

from braindecode.models import ShallowFBCSPNet as LibShallow  # noqa: E402
from shallowfbcsp import ShallowFBCSPNet as SelfShallow  # noqa: E402

N_CHANS = 8
N_TIMES = 500
N_OUT = 3
DROP = 0.5
BATCH = 4

# braindecode 扁平键 → 手写模块化键
LIB_TO_SELF: dict[str, str] = {
    "conv_time_spat.conv_time.weight": "conv_time.conv.weight",
    "conv_time_spat.conv_time.bias": "conv_time.conv.bias",
    "conv_time_spat.conv_spat.weight": "conv_spat.conv.weight",
    "bnorm.weight": "bnorm.weight",
    "bnorm.bias": "bnorm.bias",
    "bnorm.running_mean": "bnorm.running_mean",
    "bnorm.running_var": "bnorm.running_var",
    "bnorm.num_batches_tracked": "bnorm.num_batches_tracked",
    "final_layer.conv_classifier.weight": "final_layer.conv_classifier.weight",
    "final_layer.conv_classifier.bias": "final_layer.conv_classifier.bias",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _build_pair(*, seed: int) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(seed)
    lib = LibShallow(N_CHANS, N_OUT, N_TIMES, drop_prob=DROP)
    torch.manual_seed(seed + 1)
    self_m = SelfShallow(N_CHANS, N_OUT, N_TIMES, drop_prob=DROP, attn=None)
    lib.eval()
    self_m.eval()
    return lib, self_m


def _copy_lib_weights_to_self(lib: nn.Module, self_m: nn.Module) -> list[str]:
    lib_sd = lib.state_dict()
    self_sd = self_m.state_dict()
    missing: list[str] = []
    for lk, sk in LIB_TO_SELF.items():
        if lk not in lib_sd:
            missing.append(f"missing_lib:{lk}")
            continue
        if sk not in self_sd:
            missing.append(f"missing_self:{sk}")
            continue
        if lib_sd[lk].shape != self_sd[sk].shape:
            missing.append(f"shape:{lk}->{sk} {tuple(lib_sd[lk].shape)} vs {tuple(self_sd[sk].shape)}")
            continue
        self_sd[sk] = lib_sd[lk].clone()
    self_m.load_state_dict(self_sd)
    return missing


def _feature_hook_report(lib: nn.Module, self_m: nn.Module, x: torch.Tensor) -> dict:
    lib_shapes: list[tuple] = []
    self_shapes: list[tuple] = []

    def hook_lib(_m, _inp, out):
        lib_shapes.append(tuple(out.shape))

    def hook_self(_m, _inp, out):
        self_shapes.append(tuple(out.shape))

    h1 = lib.conv_time_spat.register_forward_hook(hook_lib)
    h2 = self_m.conv_spat.register_forward_hook(hook_self)
    try:
        with torch.no_grad():
            lib(x)
            self_m(x)
    finally:
        h1.remove()
        h2.remove()

    return {
        "lib_after_conv_block": lib_shapes[-1] if lib_shapes else None,
        "self_after_conv_spat": self_shapes[-1] if self_shapes else None,
    }


def _forward_diff(lib: nn.Module, self_m: nn.Module, x: torch.Tensor) -> dict:
    with torch.no_grad():
        logit_l = lib(x)
        logit_s = self_m(x)
    diff = (logit_l - logit_s).abs()
    pl = torch.softmax(logit_l, dim=-1)
    ps = torch.softmax(logit_s, dim=-1)
    pdiff = (pl - ps).abs()
    return {
        "logit_max_abs": float(diff.max()),
        "logit_mean_abs": float(diff.mean()),
        "prob_max_abs": float(pdiff.max()),
        "prob_mean_abs": float(pdiff.mean()),
        "shape_lib": list(logit_l.shape),
        "shape_self": list(logit_s.shape),
    }


def _grad_smoke(lib: nn.Module, self_m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    lib.train()
    self_m.train()
    crit = nn.CrossEntropyLoss()
    xl = x.clone().requires_grad_(False)
    xs = x.clone().requires_grad_(False)
    out_l = lib(xl)
    out_s = self_m(xs)
    loss_l = crit(out_l, y)
    loss_s = crit(out_s, y)
    loss_l.backward()
    loss_s.backward()

    def _grad_norm(m: nn.Module) -> float:
        total = 0.0
        for p in m.parameters():
            if p.grad is not None:
                total += float(p.grad.data.norm().item() ** 2)
        return total**0.5

    lib_g = _grad_norm(lib)
    self_g = _grad_norm(self_m)
    lib.eval()
    self_m.eval()
    return {
        "lib_grad_norm": lib_g,
        "self_grad_norm": self_g,
        "ratio_self_over_lib": (self_g / lib_g) if lib_g > 0 else None,
        "loss_lib": float(loss_l.item()),
        "loss_self": float(loss_s.item()),
    }


def run_audit(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lib, self_m = _build_pair(seed=42)
    x = torch.randn(BATCH, N_CHANS, N_TIMES)

    lib_keys = list(lib.state_dict().keys())
    self_keys = list(self_m.state_dict().keys())

    report: dict = {
        "stamp": _utc_stamp(),
        "config": {
            "n_chans": N_CHANS,
            "n_times": N_TIMES,
            "n_outputs": N_OUT,
            "drop_prob": DROP,
            "self_attn": None,
        },
        "A_static": {
            "lib_params": _param_count(lib),
            "self_params": _param_count(self_m),
            "params_equal": _param_count(lib) == _param_count(self_m),
            "lib_state_keys": lib_keys,
            "self_state_keys": self_keys,
            "key_map": LIB_TO_SELF,
        },
        "B1_independent_random": _forward_diff(lib, self_m, x),
        "B2_weight_copied": {},
        "B3_grad_smoke": {},
        "verdict": {},
    }

    copy_missing = _copy_lib_weights_to_self(lib, self_m)
    report["B2_weight_copied"]["copy_missing"] = copy_missing
    report["B2_weight_copied"]["forward"] = _forward_diff(lib, self_m, x)
    report["B2_weight_copied"]["feature_shapes"] = _feature_hook_report(lib, self_m, x)

    y = torch.randint(0, N_OUT, (BATCH,))
    report["B3_grad_smoke"] = _grad_smoke(lib, self_m, x, y)

    b2 = report["B2_weight_copied"]["forward"]
    b2_ok = (
        not copy_missing
        and b2["logit_max_abs"] < 1e-5
        and b2["prob_max_abs"] < 1e-5
    )
    report["verdict"] = {
        "A_params_match": report["A_static"]["params_equal"],
        "B2_numerically_equivalent": b2_ok,
        "B2_logit_max_abs": b2["logit_max_abs"],
        "recommendation": (
            "等价：可用手写 S0 替换 braindecode L0（训练环一致前提下）"
            if b2_ok
            else "需排查：权重拷贝或前向仍有差异，先修 shallowfbcsp 再开五折"
        ),
    }

    json_path = out_dir / f"compare_shallow_{report['stamp']}.json"
    md_path = out_dir / f"compare_shallow_{report['stamp']}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 方案 18 · Shallow 实现对齐审计（A/B）",
        "",
        f"- 时间：{report['stamp']}",
        f"- 参数量：lib={report['A_static']['lib_params']} self={report['A_static']['self_params']}",
        "",
        "## B2 权重拷贝后前向",
        "",
        f"- logit max|Δ| = {b2['logit_max_abs']:.3e}",
        f"- prob max|Δ| = {b2['prob_max_abs']:.3e}",
        f"- copy_missing = {copy_missing or '[]'}",
        "",
        "## 结论",
        "",
        f"- **{report['verdict']['recommendation']}**",
        "",
        f"完整 JSON：`{json_path.name}`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="方案18 · Shallow 手写 vs 库 对齐审计")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    run_audit(args.out)


if __name__ == "__main__":
    main()
