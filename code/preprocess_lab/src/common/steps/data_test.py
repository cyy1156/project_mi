from pathlib import Path
import numpy as np


def data_test() -> None:
    # 本机：仓库根 D:/cyy/MI；正式 2s 输出在 out/bci2a_2s
    out = Path("D:/cyy/MI/code/preprocess_lab/out/bci2a_2s")
    for split in ("train", "val"):
        X = np.load(out / f"{split}_X.npy")
        yt = np.load(out / f"{split}_y_task.npy")
        y3 = np.load(out / f"{split}_y_three.npy")
        print(split, X.shape, X.dtype, yt.shape, y3.shape)
        assert X.ndim == 4 and X.shape[1:] == (1, 8, 500)
        assert len(X) == len(yt) == len(y3)
        assert set(np.unique(yt)).issubset({0, 1})
        assert set(np.unique(y3)).issubset({0, 1, 2})
        # 标签一致性：three=0 ↔ task=0；three∈{1,2} → task=1
        assert np.all((y3 == 0) == (yt == 0))
        assert np.all(yt[y3 > 0] == 1)

    print("data ok")


if __name__ == "__main__":
    data_test()
