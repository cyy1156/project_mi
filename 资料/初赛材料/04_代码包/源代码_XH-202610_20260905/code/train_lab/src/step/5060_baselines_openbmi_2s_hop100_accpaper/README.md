# 5060_baselines_openbmi_2s_hop100_accpaper

OpenBMI · 2s/hop100 · **Acc_paper** 选模重训（**十一模型**）。  
**正式结果以本机 RTX 5060 · Fast 模式为准**；5090 仅对照，勿覆盖正式表。

- 方案：`资料/模型训练/04_5060_旁路_2s滑窗100ms_openbmi_accpaper/方案.md`
- 正式清单：`资料/模型训练/5060_openbmi_accpaper_实验与权重清单.md`
- 数据：`preprocess_lab/out/openbmi_2s_hop100/`（先跑预处理）
- 权重输出：`train_lab/out/5060_baseline_openbmi_2s_hop100_accpaper/`

## 双模式（方案 A）

| 模式 | 用途 | 开关 | 是否正式 |
|------|------|------|----------|
| **Fast**（默认） | 日常全量训练、出 Acc_paper 正式表 | AMP + cudnn.benchmark | **是** |
| **Repro** | 同机抽检可复现（不必与 Fast 数字相同） | `--repro` | 否 |

```bash
cd code/train_lab/src/step/5060_baselines_openbmi_2s_hop100_accpaper

# Fast：正式出数（吞吐优先）
python run_all.py --continue-on-error
python baseline_eegnet.py --batch-train 256          # 轻模型可试更大 batch

# Repro：抽检（关 AMP / benchmark，workers=0）
python baseline_eegnet.py --repro --max-folds 1      # 冒烟
python baseline_eegnet.py --repro                    # 全五折抽检（慢）
```

`--repro` ≡ `--deterministic --no-amp --num-workers 0`（并关 TF32）。

## Fast 默认（最大化训练）

| 项 | 默认 | 说明 |
|---|---|---|
| `batch_train` / `batch_eval` | 128 / 256 | 轻模型可试 `--batch-train 256`；OOM 再降 |
| `num_workers` | **2** | 内存够可 `--num-workers 4` |
| `use_amp` | True | |
| `cudnn.benchmark` | True | |
| `torch_num_threads` | 6 | 主进程线程；给 DataLoader 留核 |

## 双机同步（本机 5060）

1. `git pull --rebase`
2. 只改本包与 `04_5060_*` / `5060_*` 清单
3. `git push`

不要改 `5090_baselines_*` / `04_5090_*`。

## 预处理

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --limit 1
python -m src.datasets.openbmi.batch_2s_hop100
```

## 模型（11）

shallow · deep · conformer · eegnet · eegtcnet · gcbnet · dgcnn · dbn · dbn_raw · gcbnet_raw · dgcnn_raw
