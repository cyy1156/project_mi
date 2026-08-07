# baselines5060_openbmi_2s_hop100_accpaper

OpenBMI · 2s/hop100 · **Acc_paper** 选模重训（**十一模型**）。  
**本机：RTX 5060** — 只在本机改/推此包；5090 机请用 `5090_baselines_openbmi_2s_hop100_accpaper/`。

- 方案：`资料/模型训练/04_5060_旁路_2s滑窗100ms_openbmi_accpaper/方案.md`
- 数据：`preprocess_lab/out/openbmi_2s_hop100/`（先跑预处理）
- 超参：`patience=20` · balbatch · 无 RAP · 被试键方案 A
- 权重输出：`train_lab/out/baseline_openbmi_2s_hop100_accpaper/`

## 双机同步（本机 5060）

1. **先 `git pull --rebase`**（拉取 5090 上传的代码/结果）
2. 只改本包与 `04_5060_*` 文档
3. **再 `git push`**（上传本机训练代码）

不要改 `5090_baselines_*` / `04_旁路_*`（5090 侧）。

## 预处理

```bash
cd code/preprocess_lab
python -m src.datasets.openbmi.batch_2s_hop100 --limit 1          # 冒烟 1 文件
python -m src.datasets.openbmi.batch_2s_hop100 --subjects 01,02   # 小子集
python -m src.datasets.openbmi.batch_2s_hop100                    # 全量
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
cd code/train_lab/src/step/baselines5060_openbmi_2s_hop100_accpaper
# 冒烟
python baseline_eegnet.py --max-folds 1 --max-epochs 2 --patience 2
# 全量十一模型
python run_all.py --continue-on-error
python baseline_eegnet.py --num-workers 4 --batch-train 256
```

## 模型（11）

shallow · deep · conformer · eegnet · eegtcnet · gcbnet · dgcnn · dbn · dbn_raw · gcbnet_raw · dgcnn_raw
