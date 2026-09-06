# 教学代码示例怎么用

> 这些是**老师写的参考答案**，用来「看懂长什么样」。  
> 作业仍在 **`self_learing/`** 里**自己重写**；**不要**把本目录代码复制进仓库交差，也**不要**往现网 `experiment_game/` 里粘。

## 统一约定

| 项 | 约定 |
|----|------|
| 作业根目录 | `self_learing/`（与仓库并列或仓库内同名包均可，以你本机为准） |
| 源码 | `self_learing/src/self_learing/<文件>.py` |
| 测试 | `self_learing/tests/test_*.py` |
| 配置 | `self_learing/config/protocol.yaml`（可从原版拷数字） |
| 示例里的包名 | 文中常写 `teach_demo`；**你作业 import 一律改成 `self_learing`** |

## 建议学法

1. 打开对应 `M0x_代码示例.md`，先看 **「放到哪」** 树  
2. 再看「三句话」与完整代码块  
3. **合上示例**，在 `self_learing` 里自己敲  
4. 卡住再睁眼对照 10 行，再合上继续  
5. 在 `self_learing` 根目录跑：`pytest -q`

## 文件一览

| 文件 | 模块 | 作业主要落盘 |
|------|------|----------------|
| [M00_代码示例.md](M00_代码示例.md) | 脚手架 + smoke | `self_learing/` 目录树、`tests/test_smoke.py` |
| [M01_代码示例.md](M01_代码示例.md) | SampleSource | `channels.py` · `source_*.py` |
| [M02_代码示例.md](M02_代码示例.md) | Buffer + Bus + CSV | `eeg_bus.py` · `ring_buffer.py` · `csv_*.py` · `live_capture.py` |
| [M03_代码示例.md](M03_代码示例.md) | EventLog + alignment | `events.py` · `markers.py` · `alignment.py` |
| [M04_代码示例.md](M04_代码示例.md) | Trial FSM | `timing.py` · `trial_fsm.py` |
| [M05_代码示例.md](M05_代码示例.md) | 切窗 + zscore | `windowing.py` · `preprocess.py` |
| [M06_代码示例.md](M06_代码示例.md) | Fake 推理 + 融合 | `infer_fake.py` · `infer_fuse.py` · `inference_service.py` |
| [M07_代码示例.md](M07_代码示例.md) | F5 计分 | `readout.py` · `scoring.py` |
| [M08_代码示例.md](M08_代码示例.md) | Session 串联 | `session_dir.py` · `session_v3.py` · `tools/run_synthetic_session.py` |
| [M09_代码示例.md](M09_代码示例.md) | 落盘产品化 | `session_layout.py` · `subject_registry.py`（增强 `alignment.py`） |
| [M10_代码示例.md](M10_代码示例.md) | Leave-Next 调度 | `ft_filters.py` · `leave_next.py` · `tools/run_leave_next.py` |
| [M11_代码示例.md](M11_代码示例.md) | 端到端验收清单 | 不新增算法；验收命令与目录树 |

对应精读笔记：[`../modules/`](../modules/)。
