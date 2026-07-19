from pathlib import Path
import scipy.io
import numpy as np


def main() -> None:
    mat = scipy.io.loadmat(
        Path(r"d:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\bci2a\A01T.mat"),
        squeeze_me=True,
        struct_as_record=False,
    )
    run = mat["data"][3]
    print(run.trial[:48])  # 前 5 个 Cue 采样点
    print(run.y[:48])  # 对应类别
    print(run.X.shape)

if __name__ == "__main__":
    main()