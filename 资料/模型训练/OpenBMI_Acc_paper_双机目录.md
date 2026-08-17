# OpenBMI Acc_paper · 双机分类目录

> 整理时间：2026-08-07 · 方案16 双机补记 2026-08-16 · 方案17 双机补记 2026-08-17  
> **正式结果 = 本机 RTX 5060 · Fast 模式**；5090 仅对照（方案17 全量推荐 5090）。

| 类别 | 5060（本机 · **正式**） | 5090（**对照 · 非正式**） |
|------|-------------------------|---------------------------|
| 训练代码 | `code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper/` | `code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper/` |
| 权重 out | `code/train_lab/out/5060_baseline_openbmi_2s_hop100_accpaper/` | `code/train_lab/out/5090_baseline_openbmi_2s_hop100_accpaper/` |
| 方案文档 | [`04_5060_旁路_2s滑窗100ms_openbmi_accpaper/`](./04_5060_旁路_2s滑窗100ms_openbmi_accpaper/) | [`04_5090_旁路_2s滑窗100ms_openbmi_accpaper/`](./04_5090_旁路_2s滑窗100ms_openbmi_accpaper/) |
| 五折记录 | [`runs/5060_openbmi_accpaper/`](./runs/5060_openbmi_accpaper/) | [`runs/5090_openbmi_accpaper/`](./runs/5090_openbmi_accpaper/) |
| 结果汇总副本 | [`../实验结果/5060/openbmi滑窗_paper_acc/`](../实验结果/5060/openbmi滑窗_paper_acc/) | [`../实验结果/5090/openbmi滑窗_paper_acc/`](../实验结果/5090/openbmi滑窗_paper_acc/) |
| 清单 | [`5060_openbmi_accpaper_实验与权重清单.md`](./5060_openbmi_accpaper_实验与权重清单.md) **正式** | [`5090_openbmi_accpaper_实验与权重清单.md`](./5090_openbmi_accpaper_实验与权重清单.md) 对照 |

### 方案 16 · Shallow Three 复合损失（旁路）

| | 5060（本机 · 低内存试探） | 5090（**全量推荐**） |
|--|---------------------------|----------------------|
| 代码 | `code/train_lab/src/step/5060_three_hier_loss_accpaper/` | `code/train_lab/src/step/5090_three_hier_loss_accpaper/` |
| out | `code/train_lab/out/5060_three_hier_loss_accpaper/` | `code/train_lab/out/5090_three_hier_loss_accpaper/` |
| 文档 | [`16_5060_旁路_shallow_Three复合损失_openbmi_accpaper/`](./16_5060_旁路_shallow_Three复合损失_openbmi_accpaper/) | [`16_5090_旁路_shallow_Three复合损失_openbmi_accpaper/`](./16_5090_旁路_shallow_Three复合损失_openbmi_accpaper/) |
| 机位 | RTX 5060 Laptop · RAM ~16GB | RTX 5090 · **RAM 128GB · VRAM 32GB** |
| 推荐 | fold0 门控 | **`chain_all.py` 五折全链** |

### 方案 17 · 掩码未来双专家门控（旁路）

| | 5060（本机 · 低内存试探） | 5090（**全量推荐**） |
|--|---------------------------|----------------------|
| 代码 | `code/train_lab/src/step/5060_mask_future_dual_expert_accpaper/` | `code/train_lab/src/step/5090_mask_future_dual_expert_accpaper/` |
| out | `code/train_lab/out/5060_mask_future_dual_expert_accpaper/` | `code/train_lab/out/5090_mask_future_dual_expert_accpaper/` |
| 文档 | [`17_5060_旁路_掩码未来双专家门控_openbmi_accpaper/`](./17_5060_旁路_掩码未来双专家门控_openbmi_accpaper/) | [`17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/`](./17_5090_旁路_掩码未来双专家门控_openbmi_accpaper/) |
| 机位 | RTX 5060 Laptop · RAM ~16GB · 默认 fold0 / batch 64 | RTX 5090 · **RAM 128GB · VRAM 32GB** · 五折 |
| 推荐 | `chain_all.py` 主线门控 | **`chain_all.py` 全链五折** |
| A1+ 数据 | `preprocess_lab/out/openbmi_2s_hop100_pf1000/`（新臂，不改旧 hop100） | 同左 |

### 5060 双模式

- **Fast**（默认）：正式出数 · AMP + cudnn.benchmark · `python run_all.py`
- **Repro**：抽检 · `python baseline_*.py --repro`（非正式）

共享预处理：`code/preprocess_lab/` · `preprocess_lab/out/openbmi_2s_hop100/`（A0）；方案17 A1+ 另见 `openbmi_2s_hop100_pf1000/`。  
同步：`git pull --rebase` → 只改本机侧 → `git push`。
