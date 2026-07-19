EEG22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

from pathlib import Path
import numpy as np
import scipy.io
# from your types import ContinuousEEG
from src.types import ContinuousEEG, sanity_check

def load_bci2a_mat(mat_path: Path) -> list[ContinuousEEG]:
    """读取 BCI IV 2a Training .mat，每个带标签 run → 一条 ContinuousEEG。"""
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    subject = mat_path.stem[:3]  # A01
    out: list[ContinuousEEG] = []

    for run_id, run in enumerate(mat["data"]):
        trial = np.atleast_1d(run.trial) if run.trial is not None else np.array([])
        if trial.size == 0:
            continue  # 校准 run：无 trial，跳过

        x = np.asarray(run.X, dtype=np.float64)  # (n_times, 25) = 22EEG+3EOG
        y = np.atleast_1d(run.y).astype(int)     # 1..4
        artifacts = np.atleast_1d(run.artifacts).astype(int)
        fs = float(run.fs)

        events = np.column_stack([trial.astype(int), y])

        out.append(
            ContinuousEEG(
                subject=subject,
                session=f"run{run_id}",
                x=x[:, :22],          # 丢掉 EOG；真正 8 通道筛选在 Step3
                fs=fs,
                ch_names=list(EEG22),
                events=events,
                labels=y,
                artifacts=artifacts,
            )
        )
    return out

def main() -> None:
    # 改成你电脑上 A01T.mat 的真实路径
    mat_path = Path(r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\bci2a\A06T.mat")

    runs = load_bci2a_mat(mat_path)   # 得到 list[ContinuousEEG]
    print("有标签的 run 数量:", len(runs))
    
    for eeg in runs:                  # 每个 eeg 就是一个 ContinuousEEG 实例
        sanity_check(eeg)
        print("y unique:", np.unique(eeg.labels))


if __name__ == "__main__":
    main()