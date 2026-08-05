# 伪在线实验（游戏会话 · 自采被试）

> 更新：2026-08-05  
> 定位：在 `experiment_game` 已采会话上，按主线 **2 s / 100 ms** 做 **伪在线回放评测**。  
> 代码：`code/train_lab/src/step/game_pseudo_online_hop100/`（产物写入各臂子目录）

## 实验臂索引

| 臂 | 目录 | 状态 | 说明 |
|----|------|------|------|
| **01 不微调 / 零样本** | [`01_不微调_零样本/`](./01_不微调_零样本/) | **已冻结** | 只读加载 BCI2a Acc_paper 权重；不对游戏被试微调/重训 |
| **02 微调 · 前半训后半评** | [`02_微调_前半训后半评/`](./02_微调_前半训后半评/) | **代码已落地** | 前半全模型 FT、后半伪在线评；`game_ft_hop100_accpaper` |

**勿覆盖** 01 的 `out/` / `results/`；02 产物只写本臂目录。

## 本轮被试会话（各臂共用）

| 被试 | 会话目录 |
|------|----------|
| sub02 | `experiment_game/data/sessions/sub02_ses01_20260723_180607` |
| sub03 | `experiment_game/data/sessions/sub03_ses01_20260723_185153` |

## 快速入口

**01（已冻结）**

- 方案：[01_不微调_零样本/方案.md](./01_不微调_零样本/方案.md)
- 结果：[01_不微调_零样本/实验结果汇总.md](./01_不微调_零样本/实验结果汇总.md)

**02（代码已落地）**

- 方案：[02_微调_前半训后半评/方案.md](./02_微调_前半训后半评/方案.md)（全模型 FT · `max_epochs=300` · `patience=20`）

```bash
# 01 评测（现有）
cd code/train_lab/src/step/game_pseudo_online_hop100
python build_streams.py
python run_all.py --continue-on-error

# 02 微调
cd code/train_lab/src/step/game_ft_hop100_accpaper
python build_splits.py
python run_all.py --continue-on-error
```
