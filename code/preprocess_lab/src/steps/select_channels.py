import numpy as np
from src.io.load_bci2a_mat import load_bci2a_mat
from pathlib import Path

TARGET_CHANNELS = ["Cz", "C3", "C4", "CP3", "FC4", "FC3", "CP4", "CPz"]

def select_channels(x:np.ndarray,ch_names: list[str]) -> np.ndarray:
    """
        x: (n_times, n_ch)
        返回: (n_times, 8)，列顺序 = TARGET_CHANS
    """
    name_to_idx ={n:i for i,n in enumerate(ch_names)}
    missing =[c for c in TARGET_CHANNELS if c not in name_to_idx]
    if missing:
        raise KeyError(f"Missing channels: {missing}")
    idx = [name_to_idx[c] for c in TARGET_CHANNELS]
    return x[:,idx]


def test_select_channels() -> None:
    mat_path=Path(r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\bci2a\A01T.mat")
    runs=load_bci2a_mat(mat_path)

    eeg = runs[0]
    x8=select_channels(eeg.x,eeg.ch_names)
    assert x8.shape == (eeg.x.shape[0],8)
    for j,name in enumerate(TARGET_CHANNELS):
        old_j=eeg.ch_names.index(name)
        assert np.allclose(x8[:, j], eeg.x[:, old_j])
    print("OK order",TARGET_CHANNELS)

    try:
        select_channels(eeg.x[:, :5], eeg.ch_names[:5])
    except KeyError as e:
        print("expected:", e)

def main() -> None:
    test_select_channels()

if __name__ == "__main__":
    main()