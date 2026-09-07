"""步骤 1：数据检查。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MI_ROOT = Path(__file__).resolve().parent.parent
if str(MI_ROOT) not in sys.path:
    sys.path.insert(0, str(MI_ROOT))

from config import (  # noqa: E402
    SFREQ,
    SESSION_AMP_SKIP_UV,
    STAGES,
    WARMUP_AMP_WARN_UV,
    data_dir,
    reports_dir,
)
from preprocessing import (  # noqa: E402
    is_bad_session_file,
    list_session_files,
    load_session_csv,
    parse_class_from_stem,
    session_amplitude_stats,
)


def inspect_stage(stage_id: int) -> Path:
    stage = STAGES[stage_id]
    out_dir = reports_dir(stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpath = data_dir()
    if not dpath.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {dpath}")

    class_prefixes = list(stage.classes.keys())
    files = list_session_files(dpath, class_prefixes)

    lines: list[str] = []
    skip_files: list[str] = []
    warn_files: list[str] = []
    per_class: dict[str, dict] = {}

    lines.append(f"Stage {stage_id} ({stage.name}) 数据检查报告")
    lines.append(f"数据目录: {dpath}")
    lines.append(f"类别: {stage.classes}")
    marked_no = sorted(p.name for p in dpath.glob("*.csv") if is_bad_session_file(p))
    if marked_no:
        lines.append(f"已标记不可用 (_no): {', '.join(marked_no)}")
    lines.append("-" * 60)

    for path in files:
        cls = parse_class_from_stem(path.stem)
        assert cls is not None
        eeg, _ = load_session_csv(path)
        n_samples = eeg.shape[1]
        duration = n_samples / SFREQ
        stats = session_amplitude_stats(eeg)

        skip = stats["usable_max_abs"] > SESSION_AMP_SKIP_UV
        warn = stats["warmup_max_abs"] > WARMUP_AMP_WARN_UV
        if skip:
            skip_files.append(path.name)
        elif warn:
            warn_files.append(path.name)

        per_class.setdefault(cls, {"sessions": 0, "duration_sec": 0.0})
        if not skip:
            per_class[cls]["sessions"] += 1
            per_class[cls]["duration_sec"] += duration

        lines.append(f"\n[{path.name}]")
        lines.append(f"  标签: {cls} -> {stage.classes[cls]}")
        lines.append(f"  行数: {n_samples}, 时长: {duration:.2f} s")
        lines.append(
            f"  全段 |amp| mean={stats['mean_abs']:.2f}, max={stats['max_abs']:.2f} uV"
        )
        lines.append(
            f"  去 warmup 后 max|amp|={stats['usable_max_abs']:.2f} uV"
        )
        lines.append(f"  warmup max|amp|={stats['warmup_max_abs']:.2f} uV")
        if skip:
            lines.append(
                f"  ** SKIP ** (去 warmup 后 max|amp| > {SESSION_AMP_SKIP_UV} uV)"
            )
        elif warn:
            lines.append(f"  ** WARN ** (warmup max|amp| > {WARMUP_AMP_WARN_UV} uV)")

    lines.append("\n" + "=" * 60)
    lines.append("类别汇总（不含 skip 文件）:")
    for cls in sorted(per_class.keys(), key=lambda c: stage.classes[c]):
        info = per_class[cls]
        lines.append(
            f"  {cls}: {info['sessions']} session, "
            f"总时长 {info['duration_sec']:.1f} s"
        )

    lines.append(f"\n可用文件: {len(files) - len(skip_files)} / {len(files)}")
    if skip_files:
        lines.append(f"建议 skip: {', '.join(skip_files)}")
    if warn_files:
        lines.append(f"warmup 警告: {', '.join(warn_files)}")

    report_path = out_dir / "inspect_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "stage": stage.name,
        "files": [p.name for p in files],
        "marked_no_files": marked_no,
        "skip_files": skip_files,
        "warn_files": warn_files,
        "per_class": per_class,
    }
    (out_dir / "inspect_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\n报告已保存: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MI 数据检查")
    parser.add_argument("--stage", type=int, default=1, choices=sorted(STAGES.keys()))
    args = parser.parse_args()
    inspect_stage(args.stage)


if __name__ == "__main__":
    main()
