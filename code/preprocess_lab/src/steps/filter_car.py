import numpy as np
import mne #做脑电/脑磁等神经信号分析的 Python 库。
from src.io.load_bci2a_mat import load_bci2a_mat
from pathlib import Path
import matplotlib.pyplot as plt
from src.steps import select_channels
from src.steps.select_channels import select_channels


def car_reference(x: np.ndarray) -> np.ndarray:
    """x: (n_times, n_ch) → 每时刻减去全通道均值。"""
    return x-x.mean(axis=1,keepdims=True)

def notch_and_bandpass(x: np.ndarray,fs:float) -> np.ndarray:
    """
       Notch 50 Hz + Bandpass 8–30 Hz。
       mne.filter 期望 (n_ch, n_times)，注意转置。
    """
    data =x.T # (n_ch, n_times)需要把通道先放在前面
    data =mne.filter.notch_filter(data,Fs=fs,freqs=50.0,verbose=False)
    data =mne.filter.filter_data(
        data,sfreq=fs,l_freq=8.0,h_freq=30.0,verbose=False
    )
    return data.T

def car_then_filter(x: np.ndarray,fs: float ) -> np.ndarray:
    x=car_reference(x)
    x=notch_and_bandpass(x,fs)
    return x

def test_filter_car():
    mat_path=Path(r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\bci2a\A01T.mat")
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