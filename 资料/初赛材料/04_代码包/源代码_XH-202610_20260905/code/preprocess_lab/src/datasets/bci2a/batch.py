from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from src.common.config_load import load_config
from src.datasets.registry import as_run_list, get_loader
from src.datasets.bci2a.pipeline import (
    preprocess_run,
    preprocess_run_1s,
    preprocess_run_2s_hop100,
    preprocess_run_4s,
    sanity_check_outputs,
)
from src.common.steps.split_subjects import split_all_trials


def _pick_preprocess(cfg: dict):
    mode = str(cfg.get("window_mode", "")).strip().lower()
    win = float(cfg.get("win_sec", 2.0))
    if mode in {"slide_1s", "1s", "cue0_4_slide1s"}:
        return preprocess_run_1s, 250
    if mode in {"slide_2s_hop100", "2s_hop100", "cue0_4_slide2s_hop100"}:
        return preprocess_run_2s_hop100, 500
    if mode in {"cue0_4", "4s", "cue_0_4"} or win >= 3.9:
        return preprocess_run_4s, 1000
    return preprocess_run, 500

def collect_files(cfg:dict) -> dict:
    """根据 yaml 收集本数据集全部输入文件。"""
    if cfg.get("data_files"):
        files=[Path(p) for p in cfg["data_files"]]
    else:
        pattern =cfg["data_glob"]
        files =sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"没有匹配到数据文件，请检查 data_glob / data_files。dataset={cfg.get('dataset')}"
        )
    print(f"[{cfg['dataset']}]共{len(files)}个文件：")
    for f in files:
        print(" ",f)
    return files

def subject_id_from_path(path: Path, cfg: dict) -> str:
    mode =cfg.get("subject_from","stem3")
    if mode == "stem3":
        return path.stem[:3]# A01T → A01
    if mode == "stem":
        return path.stem
    raise ValueError(f"未知subject_from: {mode}")
def process_one_file(
            path: Path,
            cfg: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
       处理一个 .mat：内部多 run → 合并。
       返回 X, y_task, y_three, subjects, trial_id（与窗等长；非 1s 时 trial_id 为 -1）。
    """
    loader = get_loader(cfg["loader"])
    sid = subject_id_from_path(path, cfg)
    runs=as_run_list(loader(path))
    print(f"->{path.name}:subject={sid},runs={len(runs)}")

    preprocess_fn, n_times = _pick_preprocess(cfg)
    is_slide = preprocess_fn in (preprocess_run_1s, preprocess_run_2s_hop100)
    xs,yts,y3s,tids=[],[],[],[]
    add_rest=bool(cfg.get("make_rest",True))
    tid_offset = 0

    for eeg in runs:
        if is_slide:
            x, yt, y3, tid = preprocess_fn(eeg, add_rest=add_rest)
            if len(yt) == 0:
                continue
            tids.append(tid + tid_offset)
            tid_offset = int((tid + tid_offset).max()) + 1 if len(tid) else tid_offset
        else:
            x, yt, y3 = preprocess_fn(eeg, add_rest=add_rest)
            if len(yt) == 0:
                continue
            tids.append(np.full((len(yt),), -1, dtype=np.int64))
        xs.append(x)
        yts.append(yt)
        y3s.append(y3)

    if not xs:
        empty =np.zeros((0,1,8,n_times),np.float32)
        z = np.zeros((0,),np.int64)
        return empty,z,z.copy(),np.array([],dtype=object),z.copy()

    x=np.concatenate(xs, axis=0)
    y_task=np.concatenate(yts, axis=0)
    y_three=np.concatenate(y3s, axis=0)
    trial_id=np.concatenate(tids, axis=0)
    subject=np.array([sid]*len(y_task),dtype=object)
    return x,y_task,y_three,subject,trial_id

def process_whole_dataset(
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """扫配置中的全部文件并合并。"""
    Xs, yts, y3s, subs, tids = [], [], [], [], []

    for fpath in collect_files(cfg):
        X, yt, y3, sid, tid = process_one_file(fpath, cfg)
        if len(yt) == 0:
            print(f"  ⚠ 跳过（无有效试次）: {fpath.name}")
            continue
        print(
            f"  完成 {fpath.name}: X={X.shape}, "
            f"task={np.bincount(yt, minlength=2)}, "
            f"three={np.bincount(y3, minlength=3)}"
        )
        Xs.append(X)
        yts.append(yt)
        y3s.append(y3)
        subs.append(sid)
        tids.append(tid)

    if not Xs:
        raise RuntimeError("全部文件处理后没有任何有效试次")

    X = np.concatenate(Xs, axis=0)
    y_task = np.concatenate(yts, axis=0)
    y_three = np.concatenate(y3s, axis=0)
    subjects = np.concatenate(subs, axis=0)
    trial_id = np.concatenate(tids, axis=0)

    print("=" * 50)
    print(
        "合并完成:",
        "X", X.shape,
        "y_task", np.bincount(y_task, minlength=2),
        "y_three", np.bincount(y_three, minlength=3),
        "subjects", sorted(set(subjects.tolist())),
    )
    _, n_times = _pick_preprocess(cfg)
    sanity_check_outputs(X, y_task, y_three, n_times=n_times)
    return X, y_task, y_three, subjects, trial_id

def save_outputs(
    X: np.ndarray,
    y_task: np.ndarray,
    y_three: np.ndarray,
    subjects: np.ndarray,
    cfg: dict,
    trial_id: np.ndarray | None = None,
) -> Path:
    out_dir =Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[3] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag =cfg["dataset"]

    if cfg.get("save_full",True):
        np.save(out_dir / f"{tag}_X.npy", X)
        np.save(out_dir / f"{tag}_y_task.npy", y_task)
        np.save(out_dir / f"{tag}_y_three.npy", y_three)
        np.save(out_dir / f"{tag}_subjects.npy", subjects)
        if trial_id is not None and np.any(trial_id >= 0):
            np.save(out_dir / f"{tag}_trial_id.npy", trial_id)
        print("saved full →", out_dir)
    if cfg.get("save_split",True):
        parts = split_all_trials(
            X,
            y_task,
            y_three,
            val_ratio=float(cfg.get("val_ratio", 0.2)),
            seed=int(cfg.get("seed", 42)),
            subjects=subjects,
        )
        X_tr, yt_tr, y3_tr, sid_tr = parts["train"]
        X_va, yt_va, y3_va, sid_va = parts["val"]
        # 文件名与当前 train_lab 约定一致，便于直接训练
        np.save(out_dir / "train_X.npy", X_tr)
        np.save(out_dir / "train_y_task.npy", yt_tr)
        np.save(out_dir / "train_y_three.npy", y3_tr)
        np.save(out_dir / "train_subjects.npy", sid_tr)

        np.save(out_dir / "val_X.npy", X_va)
        np.save(out_dir / "val_y_task.npy", yt_va)
        np.save(out_dir / "val_y_three.npy", y3_va)
        np.save(out_dir / "val_subjects.npy", sid_va)
        print("saved split → train", X_tr.shape, "val", X_va.shape)


    return out_dir

def main() -> None:
    ap =argparse.ArgumentParser(description="整个批处理")
    ap.add_argument(
        "--cfg",
        default="config/bci2a.yaml",
        help="相对 preprocess_lab 的配置路径，或绝对路径",
    )
    args =ap.parse_args()
    cfg_path=Path(args.cfg)
    if not cfg_path.is_absolute():
        lab_root =Path(__file__).resolve().parents[3]
        cfg_path =lab_root / cfg_path
    cfg = load_config(cfg_path)
    X, y_task, y_three, subjects, trial_id = process_whole_dataset(cfg)
    save_outputs(X, y_task, y_three, subjects, cfg, trial_id=trial_id)
    print("全部完成:", cfg["dataset"], "→", cfg["out_dir"])


if __name__ == "__main__":
    main()
