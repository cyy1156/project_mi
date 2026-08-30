"""阶段 3 搬迁保真度校验：core/windowing vs 训练侧 src.common.steps 原始实现。

背景：重构前 offline/openbmi_align_cut.py 依赖外部 D:/MI/code/preprocess_lab/src/common/steps，
阶段 3 已将其 vendor 进 experiment_game/core/windowing.py（零项目依赖）。
本脚本用随机输入逐值比对同名函数，证明搬迁未改变数值行为。

用法：python -m experiment_game.tools.windowing_fidelity_check
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_GAME = Path(__file__).resolve().parents[1]                # D:/MI/experiment_game
_LAB = _GAME.parent / "code" / "preprocess_lab"            # D:/MI/code/preprocess_lab

if not _LAB.is_dir():
    print(f"[skip] 未找到训练侧实现目录: {_LAB}")
    raise SystemExit(0)

sys.path.insert(0, str(_LAB))

from experiment_game.core import windowing as new  # noqa: E402

try:
    from src.common.steps.epoch_baseline import (  # noqa: E402
        task_window_cue_0_to_4 as old_task_window,
    )
    from src.common.steps.slide_1s import (  # noqa: E402
        extract_segment_baseline as old_extract_baseline,
    )
    from src.common.steps.slide_3s_hop100 import (  # noqa: E402
        segment_to_3s_hop100_windows as old_seg_to_3s,
    )
except Exception as exc:  # noqa: BLE001
    print(f"[skip] 训练侧实现不可导入: {type(exc).__name__}: {exc}")
    raise SystemExit(0)


rng = np.random.default_rng(20260829)
FAILS: list[str] = []


def _cmp(name: str, a: object, b: object) -> None:
    ok = False
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape == b.shape:
            ok = bool(np.allclose(a, b, rtol=0, atol=0, equal_nan=True))
            if not ok:
                ok = bool(np.max(np.abs(a - b)) < 1e-12)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) == len(b):
            ok = all(
                np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)
            )
    else:
        ok = a == b
    print(f"  {'OK ' if ok else 'DIFF'}  {name}")
    if not ok:
        FAILS.append(name)


print("=== 1. 常量 ===")
print(f"  FS new={new.FS}  WIN_SEC={new.WIN_SEC}  HOP_SEC={new.HOP_SEC}  "
      f"N_TIMES={new.N_TIMES}  HOP_SAMPLES={new.HOP_SAMPLES}")
print(f"  FROZEN={new.FROZEN}")
assert new.N_TIMES == 750 and new.HOP_SAMPLES == 25, "窗常量异常"

print("\n=== 2. task_window_cue_0_to_4 ===")
x = rng.normal(0, 50, size=(8, 6000))
for cue in (500, 1000, 2500, 5500, 5900, 5999):
    _cmp(f"cue={cue}", new.task_window_cue_0_to_4(x, cue, new.FS),
         old_task_window(x, cue, new.FS))

print("\n=== 3. segment_to_3s_hop100_windows ===")
seg = rng.normal(0, 50, size=(8, 3000))
_cmp("shape/values", new.segment_to_3s_hop100_windows(seg, new.FS),
     old_seg_to_3s(seg, new.FS))

print("\n=== 4. extract_segment_baseline ===")
t_lsl = np.arange(6000) / 250.0 + 182900.0
for (a, b) in ((1000, 2000), (500, 1500), (0, 1000), (5500, 5999)):
    _cmp(f"[{a}, {b})", new.extract_segment_baseline(x, a, b, new.FS),
         old_extract_baseline(x, a, b, new.FS))

print("\n=== 5. slide_windows / to_nchw 自检 ===")
wins = new.slide_windows(x.T, fs=new.FS)
print(f"  slide_windows -> {len(wins)} 窗, 首窗形状 {wins[0].shape}")
assert wins[0].shape == (new.N_TIMES, 8), "滑窗形状异常"
# 注意：slide_windows 产出 (T, C)，而 to_nchw 约定入参为 (C, T) —— 需转置。
arr = new.to_nchw([w.T for w in wins])
print(f"  to_nchw((C,T) 入参) -> {arr.shape}")
assert arr.shape == (len(wins), 1, 8, new.N_TIMES), "NCHW 形状异常"

print("\n=== 6. n_windows_3s_hop100 ===")
for d in (4.0, 4.5, 6.5):
    print(f"  duration={d}s -> {new.n_windows_3s_hop100(d)} 窗")
assert new.n_windows_3s_hop100(4.0) == 11, "锚点值异常（应为 11）"

print("\n" + ("=" * 46))
if FAILS:
    print(f"结论: 存在 {len(FAILS)} 处数值差异 -> {FAILS}")
    raise SystemExit(1)
print("结论: core/windowing 与训练侧原始实现逐值一致 [OK]")
