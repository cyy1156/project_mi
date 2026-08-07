# 04_5060 · OpenBMI · 2s/hop100 · Acc_paper 选模（本机 RTX 5060）

> 状态：**十一模型方案** · 2026-08-05  
> 被试键：**A** · patience=20 · Task+Three · **全部 11 模型**（含 `*_raw`）· **语料=仅 EEG_MI_train**  
> **本目录由 5060 机维护并上传**；5090 机请用 `04_旁路_2s滑窗100ms_openbmi_accpaper/`。

| 文档 | 说明 |
|------|------|
| [`方案.md`](./方案.md) | 协议 + **§1.6 预处理步骤** + §3 十一模型 |
| 预处理 | `code/preprocess_lab/src/datasets/openbmi/`（共享；改前先 pull） |
| 训练（5060） | `code/train_lab/src/step/baselines5060_openbmi_2s_hop100_accpaper/` |
| 训练（5090，只读） | `code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper/` |

本机同步顺序：`git pull --rebase` → 改 5060 包/本文档 → `git push`。

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --limit 1 --reset

cd code/train_lab/src/step/baselines5060_openbmi_2s_hop100_accpaper
python run_all.py --continue-on-error
```
