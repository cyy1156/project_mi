# game_ft_hop100_accpaper

游戏会话 **全模型微调**（前半 trial 训 / 后半 trial 伪在线评）。

- 方案：`资料/伪在线实验/02_微调_前半训后半评/方案.md`
- 对齐：`baselines_2s_hop100_accpaper/`（Acc_paper 早停 · balbatch · 无 RAP）
- **`finetune_mode=full_model`**：特征层 + 分类头全部可训（禁止只训分类器）
- Init：`baseline_2s_hop100_accpaper/*_balbatch_accpaper` 五折权重
- **不修改** `01_不微调_零样本` 与 BCI2a Acc_paper 主线包

## 用法

```bash
cd code/train_lab/src/step/game_ft_hop100_accpaper

python build_splits.py
python baseline_eegnet.py --smoke --subjects sub02 --skip-three
python baseline_eegnet.py
python run_all.py --continue-on-error
python run_all.py --smoke --models eegnet,shallow
```

## 模型（Top5）

| 脚本 | 结构 |
|------|------|
| `baseline_shallow.py` | ShallowFBCSPNet |
| `baseline_deep.py` | Deep4Net-compat pool=1/1 |
| `baseline_conformer.py` | EEGConformer |
| `baseline_eegnet.py` | EEGNet |
| `baseline_eegtcnet.py` | EEGTCNet |

## 产物

- 文档：`资料/伪在线实验/02_微调_前半训后半评/out|results/`
- 权重：`train_lab/out/baseline_game_ft_hop100_accpaper/<model>_game_ft_half_balbatch_accpaper/<stamp>/<subject>/`
