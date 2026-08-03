# 图特征基线示例：DGCNN / GCBNet / DBN

> 性质：**索引文档**（细节与完整可粘贴脚本见下方三篇）  
> 对照时域基线：[`代码示例_baseline_eegnet_单模型入口.md`](./代码示例_baseline_eegnet_单模型入口.md)  
> 训练策略：[`资料/实验结果说明/训练策略_二分类与三分类独立训练.md`](../实验结果说明/训练策略_二分类与三分类独立训练.md)  
> 评估协议：[`正式评估协议_被试独立五折.md`](./正式评估协议_被试独立五折.md)  
> README：[`baselines_single/README.md`](../../code/train_lab/src/step/baselines_single/README.md)

---

## 与 EEGNet / Shallow 的关键差别

| | EEGNet 等时域基线 | 本页三模型（图/特征基线） |
|--|--|--|
| 模型输入 | `(B, 8, 500)` 时域波形 | **`(B, 8, F)` 特征立方体**（默认 F=2） |
| 盘上数据 | 直接喂 Dataset | 先 `raw_to_bandpower`：`(N,1,8,500)@250Hz` → `(N,8,2)` |
| 频带 | — | 与预处理 8–30 对齐：`(8–13),(13–30)` Hz（μ+β），**log 功率**（勿再切 δ/θ/γ） |
| Dataset | `ArrayTaskDataset` / `ArrayThreeDataset` | 脚本内 **`ArrayFeatDataset`** |
| 落地状态 | 仓库已有 `.py` | **DBN / GCBNet / DGCNN 均已落地** |

其余约定与 `baselines_single` 一致：复用 `shared_hparams`、Task→Three **独立**训练（不迁权重）、被试独立五折、`seed_everything` / DataLoader `generator` 锁种。

---

## 三篇明细（含完整 `baseline_*.py`）

| 模型 | 目标脚本 | 文档 |
|------|----------|------|
| DGCNN | `baseline_dgcnn.py` | [`代码示例_baseline_dgcnn_单模型入口.md`](./代码示例_baseline_dgcnn_单模型入口.md) |
| GCBNet | `baseline_gcbnet.py` | [`代码示例_baseline_gcbnet_单模型入口.md`](./代码示例_baseline_gcbnet_单模型入口.md) |
| DBN | `baseline_dbn.py` | [`代码示例_baseline_dbn_单模型入口.md`](./代码示例_baseline_dbn_单模型入口.md) |

各明细文内含 **一个**完整可粘贴 Python 代码块（imports → `main`），以及落地检查清单。本文不重复粘贴脚本。
