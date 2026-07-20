import numpy as np
from scipy.signal import resample

def resample_to_1000(
        x_win: np.ndarray,
        fs_in: float,
        fs_out: float=250,
) -> np.ndarray:
    """
        x_win: (n_times_in, 8)，应对应正好 4 秒。
        输出: (1000, 8)
    """
    n_out =int(4.0*fs_out) #1000
    if abs(fs_in-fs_out) <1e-6 and x_win.shape[0] ==n_out:
        return x_win.astype(np.float32)
    y =resample(x_win,n_out,axis=0)
    return np.asarray(y,dtype=np.float32)

def trial_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: (1000, 8) → 同形状，每通道独立标准化。"""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def to_model_tensor(trials: list[np.ndarray]) -> np.ndarray:
    """
    trials: 每个元素 (1000, 8)
    输出: (N, 1, 8, 1000)
    """
    arr = np.stack(trials, axis=0)      # (N, 1000, 8)
    arr = np.transpose(arr, (0, 2, 1))  # (N, 8, 1000)
    return arr[:, None, :, :].astype(np.float32)

def test_resample_to_1000():
    # 已是 250Hz / 1000
    a = np.random.randn(1000, 8)
    b = resample_to_1000(a, 250.0)
    assert b.shape == (1000, 8)

    # 模拟 1000Hz 的 4s → 4000 点
    c = np.random.randn(4000, 8)
    d = resample_to_1000(c, 1000.0)
    assert d.shape == (1000, 8)
    assert d.dtype == np.float32
    print("OK")

def test_trial_zscore():
    x = np.random.randn(1000, 8).astype(np.float64) * 50 + 10
    z = trial_zscore(x)
    assert z.shape == (1000, 8)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1, atol=1e-5)

    X = to_model_tensor([z, z])
    assert X.shape == (2, 1, 8, 1000)
    print("OK")
def main():
    test_resample_to_1000()

if __name__ == "__main__":
    main()