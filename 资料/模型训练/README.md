# 模型训练资料索引

> 更新：2026-08-09 · **入口修订：2026-08-29**  
> **线上/游戏现行**：`experiment_game` · **3s hop100 · E1f 四成员**（见 `experiment_game/docs/框架冻结确认_20260829.md`）。  
> 本目录「2s 主线」为 **训练旁路历史主线**；E1f / Leave-Next / fnz FT 证据见方案 **24–29**。部署动作以 `experiment_game` 配置为准（冻结 F22）。

## 历史主线入口（2 s / 100 ms · 请知悉已非线上默认）

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
| [`04_5060_旁路_2s滑窗100ms_openbmi_accpaper/`](./04_5060_旁路_2s滑窗100ms_openbmi_accpaper/) | OpenBMI Acc_paper（**RTX 5060**） |
| [`04_5090_旁路_2s滑窗100ms_openbmi_accpaper/`](./04_5090_旁路_2s滑窗100ms_openbmi_accpaper/) | OpenBMI Acc_paper（**RTX 5090**） |
| [`05_旁路_shallow_MI特征工程_openbmi_accpaper/`](./05_旁路_shallow_MI特征工程_openbmi_accpaper/) | Shallow MI 特征工程旁路（偏侧通道等；非正式表；**已结案阴性**） |
| [`06_旁路_可教试次_子集评估_微调_openbmi_accpaper/`](./06_旁路_可教试次_子集评估_微调_openbmi_accpaper/) | 可教试次清单 · 子集评估正式权重 · 可选子集 FT（非正式表） |
| [`07_旁路_无zscore_2s滑窗_openbmi_accpaper/`](./07_旁路_无zscore_2s滑窗_openbmi_accpaper/) | 无 z-score · 2s/hop100；代码已落地（`*_hop100_noz_accpaper`）；待全量跑数 |
| [`08_旁路_无zscore_固定窗_openbmi_accpaper/`](./08_旁路_无zscore_固定窗_openbmi_accpaper/) | 无 z-score · Cue+2–4s 固定窗；代码已落地（`*_fixed_noz_accpaper`）；待全量跑数 |
| [`OpenBMI_Acc_paper_双机目录.md`](./OpenBMI_Acc_paper_双机目录.md) | 5060 / 5090 路径对照总览 |

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
