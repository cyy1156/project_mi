# game_pseudo_online_hop100

游戏会话伪在线。当前默认臂：**不微调 / 零样本**。

- 方案：`资料/伪在线实验/01_不微调_零样本/方案.md`
- 总索引：`资料/伪在线实验/README.md`
- 目录形态对齐 `baselines_2s_hop100_accpaper/`：每模型一个 `baseline_*.py`

## 用法

```bash
cd code/train_lab/src/step/game_pseudo_online_hop100

python build_streams.py
python baseline_eegnet.py
python baseline_shallow.py --skip-three --smoke
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

权重只读：`train_lab/out/baseline_2s_hop100_accpaper/<model>_2s_hop100_balbatch_accpaper/...`  
主报：段级 Acc_paper。  
硬约束：`no_finetune`（不对游戏被试训练）。

## 产物（写入 01 臂目录）

- `资料/伪在线实验/01_不微调_零样本/out/<session>/segment_index.jsonl`
- `资料/伪在线实验/01_不微调_零样本/results/<stamp>_<model>_pseudo_online/`
