# 伪在线实验（游戏会话 · 自采被试）

> 更新：2026-08-09  
> 定位：在 `experiment_game` 已采会话上，按主线 **2 s / 100 ms** 做 **伪在线回放评测**。  
> 代码：`code/train_lab/src/step/game_pseudo_online_hop100/`（产物写入各臂子目录）

## 当前部署口径（已接受 · 2026-08-09）

| 项 | 约定 |
|----|------|
| 零样本主权重 | **OpenBMI 正式 Acc_paper shallow** `run_20260807_135828`（证据：[04](./04_旁路_OpenBMI权重_游戏零样本与门控/)） |
| 通道 | 游戏序 → OpenBMI 序（`eval_openbmi_game.py`） |
| 质量门控 | 可选叠加 **H1**（ERD≤−15 & lat≥8；同 trial REST 基线）；接受 ~16–21% 段 abstain |
| 不采纳 | BCI2a 权重上硬套 H1（[03-P2](./03_旁路_teachable质量门控/) 阴性）；06 B2 微调头作全会话主权重 |
| 历史对照 | [01](./01_不微调_零样本/) / [02](./02_微调_前半训后半评/) 数字**冻结保留**，不改写夺冠 |

## 实验臂索引

| 臂 | 目录 | 状态 | 说明 |
|----|------|------|------|
| **01 不微调 / 零样本** | [`01_不微调_零样本/`](./01_不微调_零样本/) | **已冻结** | BCI2a Acc_paper 零样本对照（历史主线） |
| **02 微调 · 前半训后半评** | [`02_微调_前半训后半评/`](./02_微调_前半训后半评/) | **正式结果已汇总** | 前半全模型 FT、后半伪在线评；`game_ft_hop100_accpaper` |
| **03 teachable 质量门控** | [`03_旁路_teachable质量门控/`](./03_旁路_teachable质量门控/) | **P1+/P2+** | P2 **现行=OpenBMI（与 04 对齐）**；历史 BCI2a-P2 阴性冻结 |
| **04 OpenBMI 权重·游戏** | [`04_旁路_OpenBMI权重_游戏零样本与门控/`](./04_旁路_OpenBMI权重_游戏零样本与门控/) | **已采纳** | 与 03-P2 同协议；部署主权重 + 可选 H1 |
| **05 OpenBMI 前半FT后半评** | [`05_旁路_OpenBMI_前半微调后半评/`](./05_旁路_OpenBMI_前半微调后半评/) | **已结案** | OpenBMI shallow 前半 FT；sub03 强 / sub02 Task 弱于 02 |
| **06 前半FT+后半门控** | [`06_旁路_OpenBMI_前半FT后半门控/`](./06_旁路_OpenBMI_前半FT后半门控/) | **已结案** | 05 FT+H1：Three+；Task sub02+/sub03−（≠模型训练06） |

**勿覆盖** 01 的 `out/` / `results/`；02–06 产物只写本臂目录。

## 本轮被试会话（各臂共用）

| 被试 | 会话目录 |
|------|----------|
| sub02 | `experiment_game/data/sessions/sub02_ses01_20260723_180607` |
| sub03 | `experiment_game/data/sessions/sub03_ses01_20260723_185153` |

## 快速入口

**01（已冻结）**

- 方案：[01_不微调_零样本/方案.md](./01_不微调_零样本/方案.md)
- 结果：[01_不微调_零样本/实验结果汇总.md](./01_不微调_零样本/实验结果汇总.md)

**02（正式结果已汇总）**

- 方案：[02_微调_前半训后半评/方案.md](./02_微调_前半训后半评/方案.md)（全模型 FT · `max_epochs=300` · `patience=20`）
- 结果：[02_微调_前半训后半评/实验结果汇总.md](./02_微调_前半训后半评/实验结果汇总.md)

**03（P1+/P2− 已结案）**

- 方案：[03_旁路_teachable质量门控/方案.md](./03_旁路_teachable质量门控/方案.md)
- 登记：[03_旁路_teachable质量门控/总结/结果登记表.md](./03_旁路_teachable质量门控/总结/结果登记表.md)

**04（OpenBMI shallow · 游戏 Q0/Q1 · 已采纳）**

- 方案：[04_旁路_OpenBMI权重_游戏零样本与门控/方案.md](./04_旁路_OpenBMI权重_游戏零样本与门控/方案.md)
- 登记：[04_旁路_OpenBMI权重_游戏零样本与门控/总结/结果登记表.md](./04_旁路_OpenBMI权重_游戏零样本与门控/总结/结果登记表.md)

**05（OpenBMI shallow · 前半 FT / 后半评）**

- 方案：[05_旁路_OpenBMI_前半微调后半评/方案.md](./05_旁路_OpenBMI_前半微调后半评/方案.md)
- 登记：[05_旁路_OpenBMI_前半微调后半评/总结/结果登记表.md](./05_旁路_OpenBMI_前半微调后半评/总结/结果登记表.md)

**06（05 FT + 后半门控）**

- 方案：[06_旁路_OpenBMI_前半FT后半门控/方案.md](./06_旁路_OpenBMI_前半FT后半门控/方案.md)
- 登记：[06_旁路_OpenBMI_前半FT后半门控/总结/结果登记表.md](./06_旁路_OpenBMI_前半FT后半门控/总结/结果登记表.md)

```bash
# 01 评测（现有）
cd code/train_lab/src/step/game_pseudo_online_hop100
python build_streams.py
python run_all.py --continue-on-error

# 02 微调（BCI2a init · 已冻结）
cd code/train_lab/src/step/game_ft_hop100_accpaper
python build_splits.py
python run_all.py --continue-on-error

# 03 P2（默认 OpenBMI · 与 04 对齐）
cd ../game_pseudo_online_hop100
python eval_gated_p2.py --model shallow --gates H0,H1,H2,H3
# 历史：python eval_gated_p2.py --weight-domain bci2a ...

# 04（同协议独立产物目录）
python eval_openbmi_game.py --model shallow --gates H0,H1,H2,H3

# 05 OpenBMI shallow 前半 FT / 后半评
cd ../game_ft_openbmi_hop100_accpaper
python build_splits.py
python baseline_shallow.py

# 06：05 FT 权重 + 后半门控
cd ../game_pseudo_online_hop100
python eval_ft_gated.py --gates H0,H1,H2,H3
```
