"""Runtime 通道特征工程：从 openbmi_2s_hop100 的 8 导窗派生差模 / Mu 包络。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

CHANS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]
IX = {n: i for i, n in enumerate(CHANS)}
FS = 250.0
_GATHER = 2048


def _as_b8t(block: np.ndarray) -> np.ndarray:
    """(B,1,8,T)|(B,8,T) → (B,8,T)。"""
    block = np.asarray(block)
    if block.ndim == 4 and block.shape[1] == 1:
        return block[:, 0, :, :]
    if block.ndim == 3 and block.shape[1] == 8:
        return block
    raise ValueError(f"unexpected block shape: {block.shape}")


def laterality_expand_b8t(x: np.ndarray) -> np.ndarray:
    """(B,8,T) → (B,10,T)：追加 C3−C4、CP3−CP4。"""
    d1 = x[:, IX["C3"], :] - x[:, IX["C4"], :]
    d2 = x[:, IX["CP3"], :] - x[:, IX["CP4"], :]
    return np.concatenate([x, d1[:, None, :], d2[:, None, :]], axis=1)


def _mu_envelope_2ch(x8: np.ndarray) -> np.ndarray:
    """(B,8,T) → (B,2,T)：C3/C4 的 8–13 Hz Hilbert 包络（窗内 z-score）。"""
    b, a = butter(4, [8.0 / (FS / 2.0), 13.0 / (FS / 2.0)], btype="band")
    out = np.empty((x8.shape[0], 2, x8.shape[2]), dtype=np.float32)
    for j, ch in enumerate(("C3", "C4")):
        sig = x8[:, IX[ch], :].astype(np.float64, copy=False)
        # filtfilt 需足够长度；2s@250=500 足够
        filt = filtfilt(b, a, sig, axis=-1)
        env = np.abs(hilbert(filt, axis=-1)).astype(np.float32)
        mu = env.mean(axis=-1, keepdims=True)
        sd = env.std(axis=-1, keepdims=True) + 1e-6
        out[:, j, :] = (env - mu) / sd
    return out


def laterality_mu_expand_b8t(x: np.ndarray) -> np.ndarray:
    """(B,8,T) → (B,12,T)：8 + 偏侧2 + Mu包络2。"""
    x10 = laterality_expand_b8t(x)
    env = _mu_envelope_2ch(x)
    return np.concatenate([x10, env], axis=1)


def materialize_expanded_mmap(
    X_src: np.ndarray,
    out_path: Path,
    *,
    mode: str,
    dtype=np.float16,
) -> np.ndarray:
    """
    将全库窗展开为 (N,1,C,T) float16 memmap 并返回（带 filename，供 DataLoader mmap）。
    mode: laterality | laterality_mu
    """
    if mode == "laterality":
        expand = laterality_expand_b8t
        n_ch = 10
    elif mode == "laterality_mu":
        expand = laterality_mu_expand_b8t
        n_ch = 12
    else:
        raise ValueError(mode)

    n = int(X_src.shape[0])
    t = int(X_src.shape[-1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        # 复用已有缓存（形状匹配才跳过）
        try:
            existing = np.load(out_path, mmap_mode="r")
            if existing.shape == (n, 1, n_ch, t):
                print(f"[channel_fe] reuse cache {out_path} {existing.shape}", flush=True)
                return existing
        except Exception:
            pass
        out_path.unlink()

    print(f"[channel_fe] writing {mode} → {out_path} shape=({n},1,{n_ch},{t})", flush=True)
    fp = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=(n, 1, n_ch, t)
    )
    for s in range(0, n, _GATHER):
        e = min(s + _GATHER, n)
        block = expand(_as_b8t(np.array(X_src[s:e])))
        fp[s:e, 0] = block.astype(dtype, copy=False)
        if (s // _GATHER) % 20 == 0:
            print(f"  … {e}/{n}", flush=True)
    fp.flush()
    del fp
    return np.load(out_path, mmap_mode="r")


def prepare_laterality_X(X: np.ndarray) -> np.ndarray:
    cache = (
        Path(__file__).resolve().parents[3]
        / "out"
        / "_fe_cache"
        / "openbmi_2s_hop100_laterality10_f16.npy"
    )
    return materialize_expanded_mmap(X, cache, mode="laterality")


def prepare_laterality_mu_X(X: np.ndarray) -> np.ndarray:
    cache = (
        Path(__file__).resolve().parents[3]
        / "out"
        / "_fe_cache"
        / "openbmi_2s_hop100_laterality_mu12_f16.npy"
    )
    return materialize_expanded_mmap(X, cache, mode="laterality_mu")
