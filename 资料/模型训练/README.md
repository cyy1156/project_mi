# 模型训练资料索引

> 更新：2026-08-04。主线已切换为 **2 s / 100 ms 滑窗 · BCI2a→OpenBMI · shallow/eegnet/conformer 深研**（严格 Val **BalAcc** 早停 + **batch balance**）。  
> 旧「1 s / 40 ms + Stieger」主线已迁入 [`归档/04_旧主线_1s伪在线/`](./归档/04_旧主线_1s伪在线/)。

## 当前主线（请从这里读）

目录：[`00_当前主线_2s滑窗100ms/`](./00_当前主线_2s滑窗100ms/)

| 文档 | 用途 |
|------|------|
| [`实验操作手册.md`](./00_当前主线_2s滑窗100ms/实验操作手册.md) | **现在怎么跑** |
| [`方案.md`](./00_当前主线_2s滑窗100ms/方案.md) | 协议冻结（2 s / 100 ms、三模型、OpenBMI、BalAcc+balbatch） |

旧路径占位（重定向）：[`00_当前主线_1s伪在线/`](./00_当前主线_1s伪在线/)

## 主线支撑结果

| 目录 | 角色 |
|------|------|
| [`01_旁路_2s滑窗100ms/`](./01_旁路_2s滑窗100ms/) | 阶段 A：窗级十一模型选型（BCI2a） |
| [`02_固定窗_bci2a_cue2to4s/`](./02_固定窗_bci2a_cue2to4s/) | 固定窗对照（非主训协议） |
| [`03_旁路_2s滑窗100ms_试次多数票/`](./03_旁路_2s滑窗100ms_试次多数票/) | 阶段 A′：试次过半投票 / Acc_paper 选模（BCI2a） |
| [`04_旁路_2s滑窗100ms_openbmi_accpaper/`](./04_旁路_2s滑窗100ms_openbmi_accpaper/) | 阶段 D：OpenBMI · Acc_paper 重训（Task Top-8；**代码已落地**） |

新实验 MD：[`runs/`](./runs/)  
最新入口：[`五折实验记录_最新.md`](./五折实验记录_最新.md)

## 归档（旧协议，勿当主结果）

目录：[`归档/`](./归档/)

| 子目录 | 内容 |
|--------|------|
| [`04_旧主线_1s伪在线/`](./归档/04_旧主线_1s伪在线/) | **已废止**的 1 s / 40 ms + Stieger 主线方案与手册 |
| [`01_计划文档_整段与特异度时代/`](./归档/01_计划文档_整段与特异度时代/) | Task 特异度、正式五折、阶段B双头等 |
| [`02_代码示例与名词/`](./归档/02_代码示例与名词/) | 单模型入口代码示例、名词对照 |
| [`03_实验结果_整段2s4s与merged/`](./归档/03_实验结果_整段2s4s与merged/) | 旧整段 / merged / 特异度臂 runs |

说明：归档数字与现行 **2 s / 100 ms** 主线 **不可混比夺冠**。

## 代码入口（当前）

| 用途 | 路径 |
|------|------|
| 2 s/hop100 预处理 BCI2a | `python -m src.datasets.bci2a.batch --cfg config/bci2a_2s_hop100.yaml` |
| 窗级训练（主线） | `code/train_lab/src/step/baselines_2s_hop100/` |
| 试次复评 | `code/train_lab/src/step/baselines_2s_hop100_trialmaj/` |
| 固定窗对照 | `code/train_lab/src/step/baselines_fixed_2s/` |
| 旧 1 s 选型（冻结） | `code/train_lab/src/step/baselines_1s/` |
| 更旧八基线（冻结） | `code/train_lab/src/step/baselines_single/` |

**已放弃**：Stieger 作为主线数据源（不再扩展主表）。  
**后续**：OpenBMI（同 2 s / 100 ms 协议接入）。
