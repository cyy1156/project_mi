# 5090_baselines_openbmi_2s_hop100_accpaper

OpenBMI · 2s/hop100 · **Acc_paper** 选模重训（全 11 模型）。  
**训练设备：NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）  
**本包由 5090 机维护并上传**；5060 机请用 `baselines5060_openbmi_2s_hop100_accpaper/`，勿改本目录。

- 方案：`资料/模型训练/04_旁路_2s滑窗100ms_openbmi_accpaper/`（5090 文档）
- 数据：`preprocess_lab/out/openbmi_2s_hop100/`（先跑预处理）
- 超参：`patience=20` · balbatch · 无 RAP · 被试键方案 A
- 权重输出：`train_lab/out/5090_baseline_openbmi_2s_hop100_accpaper/`
- 实验清单：`资料/模型训练/5090_openbmi_accpaper_实验与权重清单.md`

## 双机同步（5090 机）

1. **先 `git pull --rebase`**（拉取 5060 上传的代码）
2. 只改本包与 `04_旁路_*` / 5090 结果文档
3. **再 `git push`**

## 预处理

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --limit 1
python -m src.datasets.openbmi.batch_2s_hop100
```

## 训练

```bash
cd code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper
python baseline_eegnet.py --max-folds 1 --max-epochs 2 --patience 2
python run_all.py --continue-on-error
```

## 模型（11）

shallow · deep · conformer · eegnet · eegtcnet · gcbnet · dgcnn · dbn · dbn_raw · gcbnet_raw · dgcnn_raw
