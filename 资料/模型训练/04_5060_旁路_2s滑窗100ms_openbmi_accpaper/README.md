# 04_5060 · OpenBMI · 2s/hop100 · Acc_paper 选模（本机 RTX 5060 · **正式**）

> 状态：**十一模型方案** · 2026-08-05  
> 被试键：**A** · patience=20 · Task+Three · **全部 11 模型**（含 `*_raw`）· **语料=仅 EEG_MI_train**  
> **正式结果 = Fast 模式**；5090 仅对照。  
> 双机总览：[`../OpenBMI_Acc_paper_双机目录.md`](../OpenBMI_Acc_paper_双机目录.md)  
> 正式清单：[`../5060_openbmi_accpaper_实验与权重清单.md`](../5060_openbmi_accpaper_实验与权重清单.md)

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 协议 + **§1.6 预处理步骤** + §3 十一模型 |
| [`实验结果汇总_5060_baselines_openbmi_2s_hop100_accpaper.md`](./实验结果汇总_5060_baselines_openbmi_2s_hop100_accpaper.md) | **正式十一模型主表**（Task/Three Acc_paper） |
| 预处理 | `code/preprocess_lab/src/datasets/openbmi/`（共享；改前先 pull） |
| 训练（5060 · 正式） | `code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper/` |
| 训练（5090，对照） | `code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper/` |
| 五折记录 | `资料/模型训练/runs/5060_openbmi_accpaper/` · 结果副本 `资料/实验结果/5060/openbmi滑窗_paper_acc/` |

本机同步：`git pull --rebase` → 改 5060 包/本文档 → `git push`。

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --limit 1 --reset

cd code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper
python run_all.py --continue-on-error            # Fast 正式
python baseline_eegnet.py --repro --max-folds 1  # Repro 抽检
```
