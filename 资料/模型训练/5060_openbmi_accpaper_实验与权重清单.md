# RTX 5060 · OpenBMI Acc_paper 实验与权重清单（**正式**）

> **唯一正式表**：本机 RTX 5060 · **Fast** 模式五折结果。  
> Repro（`--repro`）仅作同机抽检，不写入本表正式排名。  
> 5090 结果见对照清单，不覆盖本表。

- 标记时间：`2026-08-07`
- 训练设备：**NVIDIA RTX 5060 Laptop**
- 训练模式：**Fast**（AMP + cudnn.benchmark；见包 README）
- 代码包：`code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper/`
- 权重根目录：`code/train_lab/out/5060_baseline_openbmi_2s_hop100_accpaper/`
- 实验记录：`资料/模型训练/runs/5060_openbmi_accpaper/`
- 结果副本：`资料/实验结果/5060/openbmi滑窗_paper_acc/`
- 方案：`资料/模型训练/04_5060_旁路_2s滑窗100ms_openbmi_accpaper/`

## 正式 run（按模型取时间最新且有「最终结论」者）

| 模型 | 状态 | 建议正式 run |
|------|------|----------------|
| eegnet | 完整 | `runs/5060_openbmi_accpaper/20260806_172218_eegnet_openbmi_2s_hop100_balbatch_accpaper` |
| shallow | 完整 | `runs/5060_openbmi_accpaper/20260807_135828_shallow_openbmi_2s_hop100_balbatch_accpaper` |
| conformer | 完整 | `runs/5060_openbmi_accpaper/20260806_132445_conformer_openbmi_2s_hop100_balbatch_accpaper` |
| eegtcnet | 完整 | `runs/5060_openbmi_accpaper/20260806_204957_eegtcnet_openbmi_2s_hop100_balbatch_accpaper` |
| dbn | 完整 | `runs/5060_openbmi_accpaper/20260805_231954_dbn_openbmi_2s_hop100_balbatch_accpaper` |
| gcbnet | 完整 | `runs/5060_openbmi_accpaper/20260806_093042_gcbnet_openbmi_2s_hop100_balbatch_accpaper` |
| dgcnn | 完整 | `runs/5060_openbmi_accpaper/20260806_111841_dgcnn_openbmi_2s_hop100_balbatch_accpaper` |
| gcbnet_raw | 完整 | `runs/5060_openbmi_accpaper/20260806_002738_gcbnet_raw_openbmi_2s_hop100_balbatch_accpaper` |
| dgcnn_raw | 完整 | `runs/5060_openbmi_accpaper/20260806_031450_dgcnn_raw_openbmi_2s_hop100_balbatch_accpaper` |
| dbn_raw | 待核验/补跑 | 最新目录可能无完整结论；建议 Fast 重跑后更新本行 |
| deep | **未完成** | `20260807_213149_...` 仅开头；需 Fast 补跑 |

## 历史 / 冒烟 run（勿当正式）

| 模型 | run |
|------|-----|
| eegnet | `20260805_181141_...` · `20260805_181355_...` |
| dbn_raw | `20260805_183346_...` · `20260805_231819_...` |

## 权重根

`code/train_lab/out/5060_baseline_openbmi_2s_hop100_accpaper/<model>_openbmi_2s_hop100_balbatch_accpaper/openbmi_2s_hop100/run_*`

## 复现抽检（非必填）

```bash
cd code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper
python baseline_eegnet.py --repro --max-folds 1
```

同机 Fast 复现判据（务实）：五折 Test Acc_paper 均值差约 &lt; 0.5～1.0 个点可接受。
