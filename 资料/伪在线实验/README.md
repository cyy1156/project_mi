# 伪在线实验（游戏会话 · 自采被试）

> 更新：2026-08-21  
> 定位：在 `experiment_game` 已采会话上做伪在线回放评测。  
> **按模型查阅**（推荐）：[`按模型/`](./按模型/)  
> 代码：`code/train_lab/src/step/game_pseudo_online_hop100/`（2s 主线；3s 见臂 07）

## 当前部署口径（2s · 已接受 · 2026-08-09）

| 项 | 约定 |
|----|------|
| 零样本主权重 | **OpenBMI 正式 Acc_paper shallow** `run_20260807_135828`（证据：[04](./04_旁路_OpenBMI权重_游戏零样本与门控/)） |
| 通道 | 游戏序 → OpenBMI 序 |
| 质量门控 | 可选叠加 **H1** |
| 历史对照 | [01](./01_不微调_零样本/) / [02](./02_微调_前半训后半评/) 数字**冻结保留** |
| **Stieger·3s 六臂复现（新）** | [07](./07_旁路_OpenBMI_3s滑窗_Stieger零样本/) · 5060 · S3·Tw=3s（待出数） |
| **Stieger·3s · 5070** | [08](./08_旁路_OpenBMI_3s滑窗_Stieger零样本_5070/) · 协议同 07 · **RTX 5070**（待 S3 权重） |

## 按模型归类

| 模型 | 索引 |
|------|------|
| **shallow** | [`按模型/shallow/`](./按模型/shallow/)（含 07/08） |
| eegnet | [`按模型/eegnet/`](./按模型/eegnet/) |
| deep | [`按模型/deep/`](./按模型/deep/) |
| conformer | [`按模型/conformer/`](./按模型/conformer/) |
| eegtcnet | [`按模型/eegtcnet/`](./按模型/eegtcnet/) |

## 实验臂索引（按协议编号）

| 臂 | 目录 | 状态 | 说明 |
|----|------|------|------|
| **01 不微调 / 零样本** | [`01_不微调_零样本/`](./01_不微调_零样本/) | **已冻结** | BCI2a Acc_paper 零样本对照 |
| **02 微调 · 前半训后半评** | [`02_微调_前半训后半评/`](./02_微调_前半训后半评/) | **已汇总** | BCI2a init · 游戏前半 FT |
| **03 teachable 质量门控** | [`03_旁路_teachable质量门控/`](./03_旁路_teachable质量门控/) | **P1+/P2+** | P2 现行=OpenBMI |
| **04 OpenBMI 权重·游戏** | [`04_旁路_OpenBMI权重_游戏零样本与门控/`](./04_旁路_OpenBMI权重_游戏零样本与门控/) | **已采纳 · 2s** | 部署主权重 |
| **05 OpenBMI 前半FT后半评** | [`05_旁路_OpenBMI_前半微调后半评/`](./05_旁路_OpenBMI_前半微调后半评/) | **已结案** | |
| **06 前半FT+后半门控** | [`06_旁路_OpenBMI_前半FT后半门控/`](./06_旁路_OpenBMI_前半FT后半门控/) | **已结案** | |
| **07 OpenBMI 3s · Stieger** | [`07_旁路_OpenBMI_3s滑窗_Stieger零样本/`](./07_旁路_OpenBMI_3s滑窗_Stieger零样本/) | **方案已立** | S3·Tw=3s·**Stieger 复现 01–06 协议**（5060） |
| **08 OpenBMI 3s · Stieger · 5070** | [`08_旁路_OpenBMI_3s滑窗_Stieger零样本_5070/`](./08_旁路_OpenBMI_3s滑窗_Stieger零样本_5070/) | **方案已立** | 协议同 07 · **RTX 5070** · 独立 out |

**勿覆盖** 01–06 的 `out/` / `results/`；07 / 08 产物只写各自目录，互不覆盖。

## 本轮被试会话（各臂共用）

| 被试 | 会话目录 |
|------|----------|
| sub02 | `experiment_game/data/sessions/sub02_ses01_20260723_180607` |
| sub03 | `experiment_game/data/sessions/sub03_ses01_20260723_185153` |

## 快速入口

**按模型（shallow / 3s）**

- 归类：[按模型/shallow/README.md](./按模型/shallow/README.md)
- 07 方案：[07_旁路_OpenBMI_3s滑窗_Stieger零样本/方案.md](./07_旁路_OpenBMI_3s滑窗_Stieger零样本/方案.md)（**Stieger** 评测集）

**04（2s 部署）**

- 方案：[04_旁路_OpenBMI权重_游戏零样本与门控/方案.md](./04_旁路_OpenBMI权重_游戏零样本与门控/方案.md)
- 登记：[04_旁路_OpenBMI权重_游戏零样本与门控/总结/结果登记表.md](./04_旁路_OpenBMI权重_游戏零样本与门控/总结/结果登记表.md)

```bash
# 04（2s · 现行）
cd code/train_lab/src/step/game_pseudo_online_hop100
python eval_openbmi_game.py --model shallow --gates H0,H1,H2,H3

# 07（3s · 脚本待接线，见该臂方案 §5）
```
