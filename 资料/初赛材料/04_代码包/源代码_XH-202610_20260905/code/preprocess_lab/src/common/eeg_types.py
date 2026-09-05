from dataclasses import dataclass
import numpy as np


@dataclass
class ContinuousEEG:
    """连续原始信号（尚未滤波、尚未切 epoch）。"""
    subject: str
    session: str
    x: np.ndarray          # (n_times, n_channels)，单位建议 μV
    fs: float              # 原始采样率
    ch_names: list[str]    # 与 x 的列一一对应
    events: np.ndarray     # (n_events, 2) → [sample_index, event_id]
    labels: np.ndarray | None = None       # 若事件自带类别可同步给出
    artifacts: np.ndarray | None = None    # 试次级伪迹，1=坏

#辅助检查
def sanity_check(eeg: ContinuousEEG) -> None:
    assert eeg.x.ndim == 2, f"x 应为 2D，实际 {eeg.x.ndim}D"
    assert eeg.x.shape[1] == len(eeg.ch_names), "通道数与 ch_names 不一致"
    assert eeg.fs > 0, "fs 必须 > 0"
    assert eeg.events.ndim == 2 and eeg.events.shape[1] == 2, "events 应为 (n, 2)"
    if eeg.labels is not None:
        assert len(eeg.labels) == len(eeg.events), "labels 与 events 长度不一致"
    print(
        eeg.subject,
        eeg.session,
        "x=", eeg.x.shape,
        "fs=", eeg.fs, "Hz",
        "events len=", len(eeg.events),
    )


def main() -> None:
    eeg = ContinuousEEG(
        subject="A01",
        session="run3",
        x=np.zeros((1000, 3)),  # 1000 个时间点，3 个通道
        fs=250.0,
        ch_names=["C3", "Cz", "C4"],
        events=np.array([[200, 1], [600, 2]]),  # 第200点左手，第600点右手
        labels=np.array([1, 2]),
        artifacts=np.array([0, 0]),
    )
    sanity_check(eeg)

if __name__ == "__main__":
    main()