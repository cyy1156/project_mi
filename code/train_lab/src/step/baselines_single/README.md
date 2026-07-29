# baselines_single

一模型一脚本：Task + Three 独立五折，写 MD；不用 registry。

| 脚本 | 模型 |
|------|------|
| `baseline_eegnet.py` | EEGNet |
| `baseline_shallow.py` | ShallowFBCSPNet |
| `baseline_deep.py` | Deep4Net |
| `baseline_eegtcnet.py` | EEGTCNet |
| `baseline_conformer.py` | EEGConformer |

约定：

- `shared_hparams.py`：全基线共用超参（含 `seed=42`）
- 权重：`code/train_lab/out/baseline/<model>/<data>/run_<stamp>/`
- MD：`资料/模型训练/runs/<stamp>_<model>/<model>五折实验记录.md`（仓库根下，不是 Desktop）
- 路径：`CODE_ROOT = HERE.parents[3]`（`code`）；勿用 `parents[4]`（会指到 `MI`，MD/权重写错处且 `import src` 失败）

## 复现性（全局锁种）

五个 `baseline_*.py` 均采用同一套思路（与 `shared_hparams.seed` 对齐）：

1. **`main` 开头**：`seed_everything(hp.seed)`  
   锁定 `random` / `numpy` / `torch` / CUDA，并设 `cudnn.deterministic=True`、`benchmark=False`。
2. **每一折建模型前**：`seed_everything(hp.seed + fold)`  
   使该折权重初始化只依赖 `seed+fold`，**不受上一折早停跑了多少 epoch 影响**（Dropout 等会消耗全局 RNG）。
3. **Train DataLoader**：`generator=make_generator(hp.seed + fold)`（仅 `shuffle=True`）  
   固定该折每个 epoch 的打乱顺序；`num_workers=0`，一般不必再设 `worker_init_fn`。
4. **被试划分**：仍把同一 `hp.seed` 传给 `iter_subject_kfold*`（内层 val 用 `seed+fold_id`）。

注意：

- `deterministic=True` 可能略慢；换 PyTorch/CUDA 版本或换卡时，仍可能有极小数值差。
- 若将来 `num_workers>0`，需为 DataLoader 增加 `worker_init_fn`。

示例：

```bash
python baseline_shallow.py --data merged_2s
python baseline_deep.py --data stieger_2s

# 可选：串跑（默认五个模型、merged_2s）
python run_all_five_model.py --data merged_2s
python run_all_five_model.py --data stieger_2s --models eegnet,shallow
```

EEGNet 文档示例：[`资料/模型训练/代码示例_baseline_eegnet_单模型入口.md`](../../../../../资料/模型训练/代码示例_baseline_eegnet_单模型入口.md)

旧 registry / matrix 入口：`../归档_旧训练入口/`。
