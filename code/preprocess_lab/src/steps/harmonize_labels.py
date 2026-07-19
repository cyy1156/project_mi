import numpy as np
from src.io.load_bci2a_mat import load_bci2a_mat
from pathlib import Path
# BCI IV 2a 原始 y: 1=左, 2=右, 3=脚, 4=舌
# → label_three: 左=1, 右=2；任务样本的 label_task 恒为 1
THREE_MAP = {1: 1, 2: 2}


def filter_left_right_events(
    events: np.ndarray,
    artifacts: np.ndarray | None,
) -> np.ndarray:
    """
    返回 shape (n_keep, 4):
      [cue_sample, label_task, label_three, trial_index]

    - 左/右均为任务态：label_task=1
    - label_three: 左手=1，右手=2
    trial_index 便于回溯原始 trial / artifacts。
    """
    kept = []
    for i, (samp, y) in enumerate(events):
        if artifacts is not None and int(artifacts[i]) == 1:
            continue
        if int(y) not in THREE_MAP:
            continue  # 脚/舌直接丢
        label_task = 1
        label_three = THREE_MAP[int(y)]
        kept.append([int(samp), label_task, label_three, i])
    if not kept:
        return np.zeros((0, 4), dtype=int)
    return np.asarray(kept, dtype=int)


def extract_rest_cues(
    cue_samples: np.ndarray,
    fs: float,
    n_times: int,
    rest_sec: float = 4.0,
    task_sec: float = 4.0,
) -> list[int]:
    """返回静息窗起点采样索引列表；标签在流水线里写成 (0, 0)。"""
    rest_len = int(rest_sec * fs)
    task_len = int(task_sec * fs)
    starts: list[int] = []
    cues = np.sort(cue_samples.astype(int))

    for i in range(len(cues) - 1):
        start = cues[i + 1] - rest_len
        end = cues[i + 1]
        prev_task_end = cues[i] + task_len

        if start < 0 or end > n_times:
            continue
        if start < prev_task_end:  # 与上一任务窗重叠
            continue
        starts.append(int(start))
    return starts

def rest_starts_to_rows(starts: list[int]) -> np.ndarray:
    """每行: [start_sample, label_task=0, label_three=0, trial_index=-1]"""
    if not starts:
        return np.zeros((0, 4), dtype=int)
    rows = [[int(s), 0, 0, -1] for s in starts]
    return np.asarray(rows, dtype=int)

def test_rest_cues() -> None:
   mat_path = Path(r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\bci2a\A01T.mat")
   runs =load_bci2a_mat(mat_path)

   eeg = runs[0]
   fs  =eeg.fs
   n_times = eeg.x.shape[0]

   kept=filter_left_right_events(eeg.events, eeg.artifacts)
   cue_samples=kept[:, 0]

   starts =extract_rest_cues(cue_samples, fs, n_times)
   rest_rows = rest_starts_to_rows(starts)

   print("===", eeg.subject, eeg.session, "===")
   print("n_left:", np.sum(kept[:, 2] == 1),
         "n_right:", np.sum(kept[:, 2] == 2))
   print("n_rest_candidates:", len(starts))
   print("前5个 rest starts:", starts[:5])
   print("换算成秒:", [s / fs for s in starts[:5]])

   # 3) 标签检查
   if len(rest_rows):
       assert np.all(rest_rows[:, 1] == 0)
       assert np.all(rest_rows[:, 2] == 0)

   # 4) 边界与重叠抽查（核心）
   cues = np.sort(cue_samples.astype(int))
   rest_len = int(4.0 * fs)
   task_len = int(4.0 * fs)

   for s in starts:
       assert 0 <= s < n_times
       assert s + rest_len <= n_times  # 4秒窗不越界

       # 找到这个静息窗对应的「下一 Cue」（窗终点）
       # 按定义：start = next_cue - 4s → next_cue = start + 4s
       next_cue = s + rest_len
       # 上一 Cue：小于 next_cue 的最大 cue
       prev_cues = cues[cues < next_cue]
       assert len(prev_cues) > 0
       prev_cue = prev_cues[-1]
       prev_task_end = prev_cue + task_len
       assert s >= prev_task_end, f"与上一任务重叠: start={s}, prev_end={prev_task_end}"

   print("rest real-data checks OK")


def test_filter_on_a01() -> None:
    mat_path = Path(r"D:\360MoveData\Users\ckgxnn\Desktop\MI\DATA\bci2a\A01T.mat")
    runs = load_bci2a_mat(mat_path)
    for eeg in runs:
        kept = filter_left_right_events(eeg.events, eeg.artifacts)
        print("===", eeg.subject, eeg.session, "===")
        print("原始试次数:", len(eeg.events))
        print("kept:", kept.shape)
        if len(kept) == 0:
            print("本 run 无左/右保留（少见）")
            continue
        print("label_task unique:", np.unique(kept[:, 1]))
        print("label_three unique:", np.unique(kept[:, 2]))
        print("n_left:", np.sum(kept[:, 2] == 1),
              "n_right:", np.sum(kept[:, 2] == 2))
        print("dropped:", len(eeg.events) - len(kept))
        assert kept.shape[1] == 4
        assert np.all(kept[:, 1] == 1)
        assert set(np.unique(kept[:, 2])).issubset({1, 2})
    print("A01 test OK")


def main() -> None:
   test_rest_cues()



if __name__ == "__main__":
    main()