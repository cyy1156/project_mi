"""步骤 2：预处理 + 数据集生成。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

MI_ROOT = Path(__file__).resolve().parent.parent
if str(MI_ROOT) not in sys.path:
    sys.path.insert(0, str(MI_ROOT))

from config import (  # noqa: E402
    ACQUISITION_FILTER,
    BALANCE_SESSIONS_PER_CLASS,
    CHANNEL_COLUMNS,
    EPOCH_SEC,
    OFFLINE_FILTER,
    PARADIGM,
    SFREQ,
    SPLIT_RANDOM_STATE,
    STAGES,
    TEST_SIZE,
    WARMUP_SEC,
    WIN_SAMPLES,
    WIN_SEC,
    WIN_STEP_SEC,
    data_dir,
    dataset_dir,
    reports_dir,
)
from data_split import split_train_test  # noqa: E402
from preprocessing import (  # noqa: E402
    balance_sessions_per_class,
    extract_windows,
    list_session_files,
    load_session_csv,
    parse_class_from_stem,
)


def _load_skip_files(stage_id: int) -> set[str]:
    meta_path = reports_dir(STAGES[stage_id]) / "inspect_meta.json"
    if not meta_path.is_file():
        return set()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return set(meta.get("skip_files", []))


def build_dataset(
    stage_id: int,
    force_skip: list[str] | None = None,
    include_all: bool = False,
    balance_sessions: bool | None = None,
) -> Path:
    stage = STAGES[stage_id]
    out_dir = dataset_dir(stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    skip: set[str] = set()
    if not include_all:
        skip = _load_skip_files(stage_id)
    if force_skip:
        skip.update(force_skip)

    dpath = data_dir()
    class_keys = list(stage.classes.keys())
    files = list_session_files(dpath, class_keys)

    balance_info: dict = {"enabled": False}
    do_balance = (
        BALANCE_SESSIONS_PER_CLASS if balance_sessions is None else balance_sessions
    )
    if do_balance:
        files, balance_info = balance_sessions_per_class(files, class_keys)
        print(
            f"类别 session 对齐: 每类使用 {balance_info['sessions_per_class']} 个 CSV"
        )
        for cls, names in balance_info["used_files"].items():
            print(f"  {cls}: {', '.join(names)}")
        for cls, names in balance_info["excluded_files"].items():
            if names:
                print(f"  {cls} 未用: {', '.join(names)}")

    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []
    all_groups: list[int] = []
    group_encoder: dict[str, int] = {}
    sessions_meta: list[dict] = []

    for path in files:
        if path.name in skip:
            print(f"跳过 {path.name}（inspect 标记 skip）")
            continue

        cls = parse_class_from_stem(path.stem)
        if cls is None or cls not in stage.classes:
            continue

        label = stage.classes[cls]
        eeg, times = load_session_csv(path)
        windows, labels, groups = extract_windows(
            eeg, path.stem, label, group_encoder
        )
        if not windows:
            print(f"警告: {path.name} 无有效窗口，已跳过")
            continue

        all_windows.extend(windows)
        all_labels.extend(labels)
        all_groups.extend(groups)
        sessions_meta.append(
            {
                "file": path.name,
                "class": cls,
                "label": label,
                "n_windows": len(windows),
                "duration_sec": eeg.shape[1] / SFREQ,
            }
        )
        print(f"{path.name}: {len(windows)} 窗口, label={label}")

    if not all_windows:
        raise RuntimeError("未生成任何样本，请检查数据或 skip 列表")

    X = np.stack(all_windows, axis=0).astype(np.float64)
    y = np.array(all_labels, dtype=np.int64)
    groups_arr = np.array(all_groups, dtype=np.int64)

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise RuntimeError(
            f"当前仅 {len(unique_labels)} 个类别（{unique_labels.tolist()}），"
            f"二分类至少需要 2 类。可用 --include-all 强制纳入 inspect 标记 skip 的文件，"
            f"或补采/更换 CSV。skip={sorted(skip)}"
        )

    train_idx, test_idx, split_info = split_train_test(
        y, groups_arr, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "groups.npy", groups_arr)
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "test_idx.npy", test_idx)

    meta = {
        "stage": stage.name,
        "stage_id": stage_id,
        "paradigm": PARADIGM,
        "acquisition_filter": ACQUISITION_FILTER,
        "offline_filter": OFFLINE_FILTER,
        "sfreq": SFREQ,
        "channels": CHANNEL_COLUMNS,
        "warmup_sec": WARMUP_SEC,
        "win_sec": WIN_SEC,
        "win_step_sec": WIN_STEP_SEC,
        "epoch_sec": EPOCH_SEC,
        "n_samples": int(X.shape[0]),
        "X_shape": list(X.shape),
        "n_groups": int(len(group_encoder)),
        "class_names": stage.class_names,
        "classes": stage.classes,
        "label_counts": dict(Counter(y.tolist())),
        "skip_files": sorted(skip),
        "sessions": sessions_meta,
        "group_map": group_encoder,
        "train_test_split": split_info,
        "session_balance": balance_info,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n数据集: {out_dir}")
    print(f"  X {X.shape}, y {y.shape}, groups {groups_arr.shape}")
    print(f"  标签分布: {meta['label_counts']}, 组数: {meta['n_groups']}")
    print(
        f"  划分: train {split_info['n_train_samples']} 样本 / "
        f"test {split_info['n_test_samples']} 样本 "
        f"({TEST_SIZE:.0%} test, 按 group)"
    )
    print(f"  train 标签: {split_info['train_label_counts']}")
    print(f"  test  标签: {split_info['test_label_counts']}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 MI 数据集")
    parser.add_argument("--stage", type=int, default=1, choices=sorted(STAGES.keys()))
    parser.add_argument("--skip", nargs="*", default=[], help="额外 skip 的文件名")
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="忽略 inspect 报告的 skip 列表，纳入全部 CSV",
    )
    parser.add_argument(
        "--no-balance-sessions",
        action="store_true",
        help="不限制每类 CSV 数量（默认每类取最少 session 数对齐）",
    )
    args = parser.parse_args()
    build_dataset(
        args.stage,
        force_skip=args.skip or None,
        include_all=args.include_all,
        balance_sessions=False if args.no_balance_sessions else None,
    )


if __name__ == "__main__":
    main()
