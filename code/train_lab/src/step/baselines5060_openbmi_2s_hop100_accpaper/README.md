# baselines_openbmi_2s_hop100_accpaper

OpenBMI · 2s/hop100 · **Acc_paper** 选模重训（**十一模型**，与 03 同名单）。

- 方案：`资料/模型训练/04_旁路_2s滑窗100ms_openbmi_accpaper/方案.md`
- 数据：`preprocess_lab/out/openbmi_2s_hop100/`（先跑预处理）
- 超参：`patience=20` · balbatch · 无 RAP · 被试键方案 A

## 预处理

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --limit 1          # 冒烟 1 文件
python -m src.datasets.openbmi.batch_2s_hop100 --subjects 01,02   # 小子集
python -m src.datasets.openbmi.batch_2s_hop100                    # 全量 108
```

## 训练

本机默认（RTX 5060 8GB / 20 核 / ~16GB RAM）已开吞吐开关：

| 项 | 默认 | 说明 |
|---|---|---|
| `batch_train` / `batch_eval` | 128 / 256 | 轻模型可试 `--batch-train 256`；OOM 再降 |
| `num_workers` | **2** | 内存够可 `--num-workers 4`；过多易 RAM OOM |
| `use_amp` | True | `--no-amp` 可关 |
| `cudnn.benchmark` | True | `--deterministic` 偏复现 |
| `torch_num_threads` | 6 | 主进程线程；给 DataLoader 留核 |

```bash
cd code/train_lab/src/step/baselines_openbmi_2s_hop100_accpaper
# 冒烟
python baseline_eegnet.py --max-folds 1 --max-epochs 2 --patience 2
# 全量十一模型
python run_all.py --continue-on-error
python baseline_eegnet.py --num-workers 4 --batch-train 256
```

## 模型（11）

| 组别 | 模型 |
|------|------|
| 时域 CNN | shallow · deep · conformer · eegnet · eegtcnet |
| bandpower 图 | gcbnet · dgcnn · dbn |
| raw + 图 | dbn_raw · gcbnet_raw · dgcnn_raw |