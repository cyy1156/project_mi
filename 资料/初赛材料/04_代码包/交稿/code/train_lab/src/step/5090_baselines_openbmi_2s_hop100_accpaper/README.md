# 5090_baselines_openbmi_2s_hop100_accpaper

OpenBMI · 2s/hop100 · **Acc_paper**（全 11 模型）。  
**训练设备：NVIDIA RTX 5090** · **本包结果为对照，非正式**。  
正式表请用本机：`5060_baselines_openbmi_2s_hop100_accpaper/` + `5060_openbmi_accpaper_实验与权重清单.md`。

- 方案：`资料/模型训练/04_5090_旁路_2s滑窗100ms_openbmi_accpaper/`
- 数据：`preprocess_lab/out/openbmi_2s_hop100/`
- 权重输出：`train_lab/out/5090_baseline_openbmi_2s_hop100_accpaper/`
- 对照清单：`资料/模型训练/5090_openbmi_accpaper_实验与权重清单.md`

## 双机同步（5090 机）

1. `git pull --rebase`（先拿 5060 正式侧更新）
2. 只改本包与 `04_5090_*`；**不要改 5060 正式清单数字**
3. `git push`

## 训练

```bash
cd code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper
python baseline_eegnet.py --max-folds 1 --max-epochs 2 --patience 2
python run_all.py --continue-on-error
```

## 模型（11）

shallow · deep · conformer · eegnet · eegtcnet · gcbnet · dgcnn · dbn · dbn_raw · gcbnet_raw · dgcnn_raw
