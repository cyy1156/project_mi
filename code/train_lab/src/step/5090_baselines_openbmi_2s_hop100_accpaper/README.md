# 5090_baselines_openbmi_2s_hop100_accpaper

OpenBMI · 2s/hop100 · **Acc_paper** 选模重训（全 11 模型）。  
**训练设备：NVIDIA RTX 5090**（32GB · sm_120 · conda `cyy` · PyTorch 2.11+cu128）

- 方案：`资料/模型训练/04_旁路_2s滑窗100ms_openbmi_accpaper/方案.md`
- 数据：`preprocess_lab/out/openbmi_2s_hop100/`（先跑预处理）
- 超参：`patience=20` · balbatch · 无 RAP · 被试键方案 A
- 权重输出：`train_lab/out/5090_baseline_openbmi_2s_hop100_accpaper/`
- 实验清单：`资料/模型训练/5090_openbmi_accpaper_实验与权重清单.md`

## 预处理

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --limit 1          # 冒烟 1 文件
python -m src.datasets.openbmi.batch_2s_hop100 --subjects 01,02   # 小子集
python -m src.datasets.openbmi.batch_2s_hop100                    # 全量 108
```

## 训练

```bash
cd code/train_lab/src/step/5090_baselines_openbmi_2s_hop100_accpaper
python baseline_eegnet.py --max-folds 1 --max-epochs 2 --patience 2
python run_all.py --continue-on-error
```

## 模型（11）

shallow · deep · conformer · eegnet · eegtcnet · gcbnet · dgcnn · dbn · dbn_raw · gcbnet_raw · dgcnn_raw

`feat_bandpower` / `load_external` / `raw_time` 复用 `baselines_2s_hop100`（经 `_hop100_path.py` append）。
