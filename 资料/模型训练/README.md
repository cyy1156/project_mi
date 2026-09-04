# 模型训练资料索引

> 更新：2026-08-09 · **入口修订：2026-08-30** · **对外命名：2026-09-04**  
> **对外模型名**：离线交卷 **QuadFold-59** · 在线主线 **CausalFuse-8** / **CausalFuse-8FT**（见 [`模型命名_QuadFold-CausalFuse.md`](./模型命名_QuadFold-CausalFuse.md)）。  
> **线上/游戏现行**：`experiment_game` · **CausalFuse-8**（3s hop100 · E1f 四成员 · T/w 冻结）· FT=all4+force → **CausalFuse-8FT**（见 `experiment_game/docs/框架冻结确认_20260829.md`）。  
> 本目录「2s 主线」为 **训练旁路历史主线**；E1f / Leave-Next / fnz FT 证据见方案 **24–31**；BCI2a 双底座×双门控见 **[`32`](./32_旁路_bci2a_LeaveNext_双底座双门控_openbmi_accpaper/)**；真被试 all4 复验见 **[`33`](./33_旁路_真被试LeaveNext_all4复验_syj_fnz_openbmi_accpaper/)**；**真人全队列统一总账见 [`41`](./41_旁路_真人被试LeaveNext_F5全队列统一_openbmi_accpaper/)**。部署动作以 `experiment_game` 配置为准（冻结 F22）。

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

## 跨域对照（常驻）

| 文档 | 用途 |
|------|------|
| [`跨域三分类成员排名对照表.md`](./跨域三分类成员排名对照表.md) | OpenBMI vs 官方指定集 · 三分类成员**名次形态**（禁混比绝对值） |
| [`指定集_Exp34-40_结果回填摘要.md`](./指定集_Exp34-40_结果回填摘要.md) | **材料回填总表** · 嵌套主读 0.540 · 折内乐观 +4.7pp · 工程/科学双轨 |
| [`35_旁路_官方vsOpenBMI_三分类排名不一致_融合重标定与骨干消融_accpaper/`](./35_旁路_官方vsOpenBMI_三分类排名不一致_融合重标定与骨干消融_accpaper/) | Exp35：融合重标定 / 三候选决赛（已结案 · S0 定稿） |
| [`36_旁路_官方主交卷_扩池与跨轨融合_accpaper/`](./36_旁路_官方主交卷_扩池与跨轨融合_accpaper/) | Exp36：**已结案** · M7/扩池/C1(45ch) 均未过 Wilcoxon 线 · 维持 S0 |
| [`37_旁路_官方主交卷_嵌套融合McNemar确认_accpaper/`](./37_旁路_官方主交卷_嵌套融合McNemar确认_accpaper/) | Exp37：**已结案** · N7 嵌套 +1.2pp / p=0.81 · **维持 S0** |
| [`38_旁路_官方主交卷_误差去相关选池_accpaper/`](./38_旁路_官方主交卷_误差去相关选池_accpaper/) | Exp38：**已结案阴性** · 嵌套贪心未成多元池 · 维持 S0 |
| [`39_旁路_官方主交卷_收尾回放与工程选卷_accpaper/`](./39_旁路_官方主交卷_收尾回放与工程选卷_accpaper/) | Exp39：**已结案** · R-B8 nested=0.540 · 科学 KEEP_S0 · 工程选 R-B8（非显著） |
| [`40_旁路_官方主交卷_CSV加固_边际校正与TTA_accpaper/`](./40_旁路_官方主交卷_CSV加固_边际校正与TTA_accpaper/) | Exp40：**已结案** · MC/TTA 阴性 · §5 曾选 R-B8 · **交卷风险否决→S0** · 算法冻结 |
| [`41_旁路_真人被试LeaveNext_F5全队列统一_openbmi_accpaper/`](./41_旁路_真人被试LeaveNext_F5全队列统一_openbmi_accpaper/) | Exp41：**结构统一** · 15 人 all4·F5 总账（自采监测；不改官方交卷） |

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
