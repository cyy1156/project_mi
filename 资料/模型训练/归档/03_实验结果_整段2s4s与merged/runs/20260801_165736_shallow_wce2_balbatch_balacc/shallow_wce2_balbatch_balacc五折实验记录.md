# 被试独立五折实验记录（20260801_165736 / shallow_wce2_balbatch_balacc）

- 开始：`2026-08-01T16:57:36`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`shallow_wce2_balbatch_balacc`（新脚本 baseline_shallow_balbatch.py；不改 baseline_shallow.py）
- 臂：`B2` — 加权CE w0=2,w1=1 + train batch balance 1:1 + BalAcc早停
- Task 采样：train WeightedRandomSampler（类逆频，约 1:1）；val/test 真实比例
- 早停：Balanced Acc；仅跑 Task（无 Three）
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balbatch_balacc\merged_2s\run_20260801_165736`

---
## 最终结论

### Task（静息/任务）— 臂 B2
- Val BalAcc（选模）：`0.5615 ± 0.0171`
- Test Spec：`0.7437 ± 0.1314`
- Test Rec：`0.3534 ± 0.1443`
- Test BalAcc：`0.5485 ± 0.0305`
- Test F1（附报）：`0.4661 ± 0.1401`
- Test Acc：`0.4618 ± 0.0703`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优（Balanced Acc）并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`68`
- 验证最优轮次（best_epoch）：`50`
- Val 选模分数（Balanced Acc）：`0.5783`
- Val F1（最优 checkpoint 时，附报）：`0.4199`
- Val loss（最优时）：`0.7871`

**Test（overall）**
- Accuracy：`0.4169`
- Recall：`0.2548`
- Specificity：`0.8175`
- Precision：`0.7752`
- F1：`0.3835`
- Balanced Acc：`0.5361`
- 混淆矩阵：TP=`1286` TN=`1671` FP=`373` FN=`3762`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5278` F1=`0.4877` BalAcc=`0.6318`
- `stieger_only`：Acc=`0.4104` F1=`0.3775` BalAcc=`0.5301`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5648`
- Val F1（最优 checkpoint 时，附报）：`0.5723`
- Val loss（最优时）：`0.6938`

**Test（overall）**
- Accuracy：`0.3490`
- Recall：`0.1402`
- Specificity：`0.8813`
- Precision：`0.7506`
- F1：`0.2362`
- Balanced Acc：`0.5107`
- 混淆矩阵：TP=`611` TN=`1507` FP=`203` FN=`3748`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3844` F1=`0.1695` BalAcc=`0.5426`
- `stieger_only`：Acc=`0.3465` F1=`0.2403` BalAcc=`0.5078`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`88`
- 验证最优轮次（best_epoch）：`70`
- Val 选模分数（Balanced Acc）：`0.5312`
- Val F1（最优 checkpoint 时，附报）：`0.3026`
- Val loss（最优时）：`0.8411`

**Test（overall）**
- Accuracy：`0.4886`
- Recall：`0.3701`
- Specificity：`0.8368`
- Precision：`0.8696`
- F1：`0.5193`
- Balanced Acc：`0.6035`
- 混淆矩阵：TP=`1760` TN=`1354` FP=`264` FN=`2995`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6487` F1=`0.7015` BalAcc=`0.6666`
- `stieger_only`：Acc=`0.4782` F1=`0.5060` BalAcc=`0.6014`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`85`
- 验证最优轮次（best_epoch）：`67`
- Val 选模分数（Balanced Acc）：`0.5569`
- Val F1（最优 checkpoint 时，附报）：`0.4687`
- Val loss（最优时）：`0.7950`

**Test（overall）**
- Accuracy：`0.5459`
- Recall：`0.5518`
- Specificity：`0.5327`
- Precision：`0.7261`
- F1：`0.6270`
- Balanced Acc：`0.5422`
- 混淆矩阵：TP=`3128` TN=`1345` FP=`1180` FN=`2541`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3913` F1=`0.3412` BalAcc=`0.4778`
- `stieger_only`：Acc=`0.5532` F1=`0.6371` BalAcc=`0.5448`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5766`
- Val F1（最优 checkpoint 时，附报）：`0.5127`
- Val loss（最优时）：`0.7153`

**Test（overall）**
- Accuracy：`0.5085`
- Recall：`0.4501`
- Specificity：`0.6500`
- Precision：`0.7569`
- F1：`0.5645`
- Balanced Acc：`0.5500`
- 混淆矩阵：TP=`2653` TN=`1582` FP=`852` FN=`3241`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3353` F1=`0.1120` BalAcc=`0.5106`
- `stieger_only`：Acc=`0.5121` F1=`0.5706` BalAcc=`0.5506`

### 实验超参
```json
{
  "data_tag": "merged_2s",
  "n_folds": 5,
  "val_ratio": 0.2,
  "seed": 42,
  "max_epochs": 300,
  "patience": 18,
  "batch_train": 32,
  "batch_eval": 64,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "drop_prob": 0.5,
  "arm": "B2",
  "task_sampler": "balanced_invfreq",
  "task_weight_mode": "fixed",
  "task_w0": 2.0,
  "task_w1": 1.0,
  "task_early_stop": "balanced_accuracy",
  "task_only": true
}
```

- 结束：`2026-08-01T17:09:03`
