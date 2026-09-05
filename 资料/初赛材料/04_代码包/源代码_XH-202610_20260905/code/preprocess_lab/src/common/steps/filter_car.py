import numpy as np
import mne #做脑电/脑磁等神经信号分析的 Python 库。
from src.datasets.bci2a.load_mat import load_bci2a_mat
from pathlib import Path
import matplotlib.pyplot as plt
from src.common.steps.select_channels import select_channels


def car_reference(x: np.ndarray) -> np.ndarray:
    """x: (n_times, n_ch) → 每时刻减去全通道均值。"""
    return x-x.mean(axis=1,keepdims=True)


def _odd_fir_len_cap(n_times: int) -> int:
    """不超过信号长度的奇数 FIR 长度。"""
    fl = int(n_times) - 1
    if fl % 2 == 0:
        fl -= 1
    return max(fl, 11)


def _adaptive_notch_kwargs(n_times: int, fs: float) -> dict:
    """
    短试次：缩短 filter_length，并加宽 trans_bandwidth，使 FIR 可设计。

    经验：fs=1000 时，默认 notch（trans_bandwidth=1）约需 6601 点；
    若信号更短，需 ``trans_bandwidth >= need / fir_len``（并留一点余量）。
    长信号仍用 MNE 默认 ``filter_length='auto'``。
    """
    if n_times <= 0:
        raise ValueError(f"n_times must be positive, got {n_times}")
    need_at_tb1 = int(round(6.601 * float(fs)))  # ≈ 默认核长
    if n_times > need_at_tb1:
        return {"filter_length": "auto"}
    fl = _odd_fir_len_cap(n_times)
    # MNE notch：参数 trans_bandwidth=T 时，有效过渡≈T/4，所需长度≈need_at_tb1/T
    tb = 1.05 * (need_at_tb1 / float(fl))
    tb = float(max(tb, 2.0))  # 短试次至少 2 Hz，保证可设计
    return {"filter_length": fl, "trans_bandwidth": tb}


def _adaptive_bandpass_kwargs(n_times: int, fs: float) -> dict:
    """长信号 auto；短试次仅封顶 filter_length（带通过渡带较宽，一般可设计）。"""
    need_at_tb1 = int(round(6.601 * float(fs)))
    if n_times > need_at_tb1:
        return {"filter_length": "auto"}
    return {"filter_length": _odd_fir_len_cap(n_times)}


def notch_and_bandpass(
    x: np.ndarray,
    fs: float,
    *,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> np.ndarray:
    """
    Notch 50 Hz + Bandpass（默认 8–30 Hz；方案19 可用 8–13 / 13–30）。
    mne.filter 期望 (n_ch, n_times)，注意转置。
    短试次按信号自适应 FIR 长度（并加宽 notch 过渡带），消除过长警告。
    """
    lo, hi = float(l_freq), float(h_freq)
    if not (lo < hi):
        raise ValueError(f"bandpass requires l_freq < h_freq, got {lo}–{hi}")
    data = x.T  # (n_ch, n_times)
    n_times = int(data.shape[-1])
    notch_kw = _adaptive_notch_kwargs(n_times, fs)
    bp_kw = _adaptive_bandpass_kwargs(n_times, fs)
    data = mne.filter.notch_filter(
        data, Fs=fs, freqs=50.0, verbose=False, **notch_kw
    )
    data = mne.filter.filter_data(
        data,
        sfreq=fs,
        l_freq=lo,
        h_freq=hi,
        verbose=False,
        **bp_kw,
    )
    return data.T

def car_then_filter(x: np.ndarray,fs: float ) -> np.ndarray:
    x=car_reference(x)
    x=notch_and_bandpass(x,fs)
    return x

def test_filter_car():
    mat_path=Path(r"D:\cyy\MI\DATA\bci2a\A01T.mat")
    runs=load_bci2a_mat(mat_path)

    eeg=runs[0]
    x8=select_channels(eeg.x,eeg.ch_names)
    x_car=car_reference(x8)
    assert np.allclose(x_car.mean(axis=1),0,atol=1e-6)

    x_f =notch_and_bandpass(x_car,eeg.fs)
    assert x_f.shape == x8.shape
    assert np.isfinite(x_f).all()

    fs = eeg.fs
    n = int(5 * fs)  # 画前 5 秒
    t = np.arange(n) / fs

    plt.plot(t, x_car[:n, 1], label="before filter (C3)")
    plt.plot(t, x_f[:n, 1], label="after filter (C3)")
    plt.xlabel("time (s)")
    plt.legend()
    plt.show(block=False)
    plt.pause(10)  # 显示几秒
    plt.close()

def main():
    test_filter_car()
if __name__ == "__main__":
    main()