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

- `shared_hparams.py`：全基线共用超参
- 权重：`code/train_lab/out/baseline/<model>/<data>/run_<stamp>/`
- MD：`资料/模型训练/runs/<stamp>_<model>/五折实验记录.md`（仓库根下，不是 Desktop）
- 路径：`CODE_ROOT = HERE.parents[3]`（`code`）；勿用 `parents[4]`（会指到 `MI`，MD/权重写错处且 `import src` 失败）

示例：

```bash
python baseline_shallow.py --data merged_2s
python baseline_deep.py --data stieger_2s
```

EEGNet 文档示例：[`资料/模型训练/代码示例_baseline_eegnet_单模型入口.md`](../../../../../资料/模型训练/代码示例_baseline_eegnet_单模型入口.md)

旧 registry / matrix 入口：`../归档_旧训练入口/`。
