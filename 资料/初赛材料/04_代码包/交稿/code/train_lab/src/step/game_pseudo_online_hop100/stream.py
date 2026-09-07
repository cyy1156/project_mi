"""切 MI/REST 段 → 同被试拼接 → 段内 2s/hop100 合法窗。"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    BASELINE_SEC,
    DOCS_OUT,
    HOP_SAMPLES,
    MIN_SEG_SEC,
    N_TIMES,
    REPO_ROOT,
    SFREQ,
    WIN_SEC,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_game.offline.epochs import (  # noqa: E402
    cut_window_with_baseline,
    resample_to_1000,
    trial_zscore,
)
from experiment_game.offline.filters import car_then_filter  # noqa: E402
from experiment_game.offline.load_session import (  # noqa: E402
    load_session,
    rejected_trial_ids,
)


@dataclass
class SegmentMeta:
    subject_id: str
    trial_id: int
    seg: str  # mi | rest
    y_three: int
    y_task: int
    t0_lsl: float
    t1_lsl: float
    n_samples: int
    stream_start: int
    stream_end: int  # exclusive


@dataclass
class EvalStream:
    subject_id: str
    session_id: str
    session_dir: str
    X: np.ndarray  # (N,1,8,500)
    y_task: np.ndarray
    y_three: np.ndarray
    trial_ids: np.ndarray
    segs: np.ndarray
    seg_keys: np.ndarray
    window_stream_starts: np.ndarray
    segments: list[SegmentMeta]
    meta: dict[str, Any]


def _require_verify(session_dir: Path) -> None:
    rep = session_dir / "alignment" / "verify_report.json"
    if not rep.is_file():
        raise FileNotFoundError(rep)
    data = json.loads(rep.read_text(encoding="utf-8"))
    if not data.get("passed"):
        raise RuntimeError(f"alignment 未通过: {rep} errors={data.get('errors')}")


def _read_trial_table(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "alignment" / "trial_table.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect_acquire_segments(
    session_dir: Path,
    *,
    subject_id: str,
    x_filt: np.ndarray,
    lsl_time: np.ndarray,
    fs: float,
    event_reject_ids: set[int] | None = None,
) -> tuple[list[np.ndarray], list[SegmentMeta], list[dict[str, Any]]]:
    rows = _read_trial_table(session_dir)
    skipped: list[dict[str, Any]] = []
    chunks: list[np.ndarray] = []
    metas: list[SegmentMeta] = []
    stream_pos = 0
    event_reject_ids = set(event_reject_ids or ())

    acquire = [
        r
        for r in rows
        if str(r.get("phase", "")).strip().lower() == "acquire"
        and int(float(r.get("rejected", 0) or 0)) == 0
        and int(r["trial_id"]) not in event_reject_ids
    ]
    acquire.sort(key=lambda r: int(r["trial_id"]))
    min_n = int(round(MIN_SEG_SEC * fs))

    for r in acquire:
        tid = int(r["trial_id"])
        label = int(float(r["label"]))
        for seg, t0_k, t1_k, y3, yt in (
            ("mi", "t_mi_start", "t_mi_end", label, 1),
            ("rest", "t_rest_start", "t_rest_end", 0, 0),
        ):
            if seg == "mi" and y3 not in (1, 2):
                skipped.append({"trial_id": tid, "seg": seg, "reason": "bad_mi_label"})
                continue
            if seg == "rest":
                y3, yt = 0, 0
            t0 = float(r[t0_k])
            t1 = float(r[t1_k])
            i0 = int(np.searchsorted(lsl_time, t0, side="left"))
            i1 = int(np.searchsorted(lsl_time, t1, side="left"))
            i0 = max(0, min(i0, x_filt.shape[0] - 1))
            i1 = max(i0 + 1, min(i1, x_filt.shape[0]))
            n = i1 - i0
            if n < min_n:
                skipped.append(
                    {"trial_id": tid, "seg": seg, "reason": "too_short", "n": int(n)}
                )
                continue
            chunk = np.asarray(x_filt[i0:i1, :], dtype=np.float64)
            meta = SegmentMeta(
                subject_id=subject_id,
                trial_id=tid,
                seg=seg,
                y_three=int(y3),
                y_task=int(yt),
                t0_lsl=t0,
                t1_lsl=t1,
                n_samples=int(chunk.shape[0]),
                stream_start=stream_pos,
                stream_end=stream_pos + int(chunk.shape[0]),
            )
            chunks.append(chunk)
            metas.append(meta)
            stream_pos += int(chunk.shape[0])
    return chunks, metas, skipped


def slide_in_segments(
    chunks: list[np.ndarray],
    metas: list[SegmentMeta],
    *,
    fs: float,
    zscore_windows: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    xs: list[np.ndarray] = []
    y_task: list[int] = []
    y_three: list[int] = []
    trial_ids: list[int] = []
    segs: list[str] = []
    seg_keys: list[str] = []
    stream_starts: list[int] = []

    n_win = int(round(WIN_SEC * fs))
    hop = max(1, int(round(HOP_SAMPLES * (fs / SFREQ))))

    for chunk, meta in zip(chunks, metas):
        t = 0
        while t + n_win <= chunk.shape[0]:
            win = cut_window_with_baseline(
                chunk, t, fs, dur_s=WIN_SEC, baseline_s=BASELINE_SEC
            )
            if win is None:
                break
            win = resample_to_1000(win, fs, fs_out=SFREQ, win_sec=WIN_SEC)
            if win.shape != (N_TIMES, 8):
                t += hop
                continue
            if zscore_windows:
                win = trial_zscore(win)
            xs.append(win)
            y_task.append(meta.y_task)
            y_three.append(meta.y_three)
            trial_ids.append(meta.trial_id)
            segs.append(meta.seg)
            seg_keys.append(f"{meta.trial_id}:{meta.seg}")
            stream_starts.append(meta.stream_start + t)
            t += hop

    empty_i = np.zeros((0,), dtype=np.int64)
    empty_o = np.zeros((0,), dtype=object)
    if not xs:
        return (
            np.zeros((0, 1, 8, N_TIMES), np.float32),
            empty_i,
            empty_i.copy(),
            empty_i.copy(),
            empty_o,
            empty_o.copy(),
            empty_i.copy(),
        )

    arr = np.stack(xs, axis=0)
    arr = np.transpose(arr, (0, 2, 1))[:, None, :, :].astype(np.float32)
    return (
        arr,
        np.asarray(y_task, dtype=np.int64),
        np.asarray(y_three, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
        np.asarray(segs, dtype=object),
        np.asarray(seg_keys, dtype=object),
        np.asarray(stream_starts, dtype=np.int64),
    )


def build_eval_stream(
    session_dir: Path | str,
    *,
    apply_filter: bool = True,
    zscore_windows: bool = True,
) -> EvalStream:
    session_dir = Path(session_dir)
    _require_verify(session_dir)
    session = load_session(session_dir)
    subject_id = session.meta.get("subject_id")
    if not subject_id:
        raise RuntimeError(f"session.meta.json 缺少 subject_id: {session_dir}")
    subject_id = str(subject_id)
    session_id = str(session.meta.get("session_id") or "")
    if abs(float(session.fs) - SFREQ) > 1e-6:
        # 仍允许非 250，但窗会 resample；主方案期望 250
        pass
    x = session.x
    if apply_filter:
        x = car_then_filter(x, session.fs)

    ev_rej = rejected_trial_ids(session.events)
    chunks, metas, skipped = collect_acquire_segments(
        session_dir,
        subject_id=subject_id,
        x_filt=x,
        lsl_time=session.lsl_time,
        fs=session.fs,
        event_reject_ids=ev_rej,
    )
    for m in metas:
        if m.subject_id != subject_id:
            raise RuntimeError(
                f"被试不一致: segment={m.subject_id} session={subject_id}"
            )
    X, yt, y3, tids, segs, keys, wstarts = slide_in_segments(
        chunks, metas, fs=session.fs, zscore_windows=zscore_windows
    )
    return EvalStream(
        subject_id=subject_id,
        session_id=session_id,
        session_dir=str(session_dir),
        X=X,
        y_task=yt,
        y_three=y3,
        trial_ids=tids,
        segs=segs,
        seg_keys=keys,
        window_stream_starts=wstarts,
        segments=metas,
        meta={
            "preprocess": "game_phase4_like",
            "filter": "CAR+notch50+bp8-30" if apply_filter else "none",
            "zscore_windows": bool(zscore_windows),
            "win_sec": WIN_SEC,
            "hop_samples": HOP_SAMPLES,
            "n_segments": len(metas),
            "n_windows": int(X.shape[0]),
            "n_mi": sum(1 for m in metas if m.seg == "mi"),
            "n_rest": sum(1 for m in metas if m.seg == "rest"),
            "skipped": skipped,
            "event_reject_ids": sorted(ev_rej),
            "phases": "acquire",
            "concat": "per_trial_mi_then_rest",
            "subject_id": subject_id,
            "fs": float(session.fs),
            "channels": list(session.ch_names),
            "win_per_full_4s_theory": 1
            + int((4.0 - WIN_SEC) / (HOP_SAMPLES / SFREQ)),
            "split_note": "game_session_all_test; no_game_train_val; bci2a_folds=weight_only",
        },
    )


def save_stream_artifacts(stream: EvalStream, out_dir: Path | None = None) -> Path:
    sid = Path(stream.session_dir).name
    out_dir = out_dir or (DOCS_OUT / "out" / sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_windows.npy", stream.X)
    np.save(out_dir / "y_task.npy", stream.y_task)
    np.save(out_dir / "y_three.npy", stream.y_three)
    np.save(out_dir / "trial_ids.npy", stream.trial_ids)
    np.save(out_dir / "seg_keys.npy", stream.seg_keys)
    with (out_dir / "segment_index.jsonl").open("w", encoding="utf-8") as f:
        for m in stream.segments:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
    (out_dir / "stream_meta.json").write_text(
        json.dumps(stream.meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return out_dir
