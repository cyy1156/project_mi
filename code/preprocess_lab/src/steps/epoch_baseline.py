import numpy as np

def slice_epoch(x: np.ndarray, cue:int ,fs :float) -> np.ndarray | None:
    """返回 (n_times_epoch, n_ch)，含基线段；越界返回 None。"""
    t0 =cue +int(-0.5*fs)
    t1 =cue +int(4.0*fs)
    if t0<0 or t1>x.shape[0]:
        return None
    return x[t0:t1,:]

def baseline_correct(epoch:np.ndarray ,fs :float) -> np.ndarray |None:
    """epoch 从 -0.5s 开始；用前 0.5s 均值归零。"""
    b1 =int(0.5*fs)
    base =epoch[:b1,:].mean(axis=0,keepdims=True)
    return epoch-base

def classification_window(epoch:np.ndarray ,fs :float) -> np.ndarray:
    """去掉基线段，只留 Cue 后 0~4s。"""
    c0 =int (0.5*fs)
    return epoch[c0:,:]

def epoch_to_class_window(x:np.ndarray,cue:int ,fs :float) -> np.ndarray| None:
    ep=slice_epoch(x,cue,fs)
    if ep is None:
        return None
    ep=baseline_correct(ep,fs)
    return classification_window(ep,fs)

def rest_window_with_baseline(x: np.ndarray, start: int, fs: float) -> np.ndarray | None:
    """截 [start, start+4s)，用开头 0.5s 均值减全窗，长度仍为 4s。"""
    n = int(4.0 * fs)
    if start < 0 or start + n > x.shape[0]:
        return None
    win = x[start:start + n, :].copy()
    b = int(0.5 * fs)
    win = win - win[:b, :].mean(axis=0, keepdims=True)
    return win

def test_baseline():
    # 假数据：fs=250，cue=125 → 前 0.5s 全是 10，后面全是 20
    x = np.zeros((2000, 8))
    x[0:125, :] = 10
    x[125:1125, :] = 20
    win = epoch_to_class_window(x, cue=125, fs=250.0)
    assert win is not None
    assert win.shape[0] == 1000
    # 基线用前 0.5s(=10) 减完后，分类窗应约为 10
    assert np.allclose(win.mean(), 10, atol=1e-6)
    print("OK")

def main() -> None:
    test_baseline()

if __name__ == "__main__":
    main()