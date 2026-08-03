# 被试独立五折实验记录（20260801_164852 / shallow_balbatch_balacc）

- 开始：`2026-08-01T16:48:52`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`shallow_balbatch_balacc`（新脚本 baseline_shallow_balbatch.py；不改 baseline_shallow.py）
- 臂：`B1` — 普通CE + train batch balance 1:1 + BalAcc早停
- Task 采样：train WeightedRandomSampler（类逆频，约 1:1）；val/test 真实比例
- 早停：Balanced Acc；仅跑 Task（无 Three）
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_balbatch_balacc\merged_2s\run_20260801_164852`

---
## 最终结论

### Task（静息/任务）— 臂 B1
- Val BalAcc（选模）：`0.5814 ± 0.0216`
- Test Spec：`0.4942 ± 0.1294`
- Test Rec：`0.6392 ± 0.1113`
- Test BalAcc：`0.5667 ± 0.0477`
- Test F1（附报）：`0.6885 ± 0.0702`
- Test Acc：`0.5966 ± 0.0611`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优（Balanced Acc）并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val 选模分数（Balanced Acc）：`0.6176`
- Val F1（最优 checkpoint 时，附报）：`0.6609`
- Val loss（最优时）：`0.6925`

**Test（overall）**
- Accuracy：`0.5402`
- Recall：`0.5406`
- Specificity：`0.5391`
- Precision：`0.7434`
- F1：`0.6260`
- Balanced Acc：`0.5399`
- 混淆矩阵：TP=`2729` TN=`1102` FP=`942` FN=`2319`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6364` F1=`0.7120` BalAcc=`0.6202`
- `stieger_only`：Acc=`0.5345` F1=`0.6208` BalAcc=`0.5352`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5587`
- Val F1（最优 checkpoint 时，附报）：`0.7747`
- Val loss（最优时）：`0.6285`

**Test（overall）**
- Accuracy：`0.5292`
- Recall：`0.4838`
- Specificity：`0.6450`
- Precision：`0.7765`
- F1：`0.5962`
- Balanced Acc：`0.5644`
- 混淆矩阵：TP=`2109` TN=`1103` FP=`607` FN=`2250`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5050` F1=`0.4451` BalAcc=`0.6197`
- `stieger_only`：Acc=`0.5309` F1=`0.6042` BalAcc=`0.5584`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5610`
- Val F1（最优 checkpoint 时，附报）：`0.6117`
- Val loss（最优时）：`0.7158`

**Test（overall）**
- Accuracy：`0.6906`
- Recall：`0.7228`
- Specificity：`0.5958`
- Precision：`0.8401`
- F1：`0.7771`
- Balanced Acc：`0.6593`
- 混淆矩阵：TP=`3437` TN=`964` FP=`654` FN=`1318`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7359` F1=`0.8209` BalAcc=`0.6496`
- `stieger_only`：Acc=`0.6876` F1=`0.7740` BalAcc=`0.6626`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5791`
- Val F1（最优 checkpoint 时，附报）：`0.5929`
- Val loss（最优时）：`0.7302`

**Test（overall）**
- Accuracy：`0.5825`
- Recall：`0.6670`
- Specificity：`0.3929`
- Precision：`0.7115`
- F1：`0.6885`
- Balanced Acc：`0.5299`
- 混淆矩阵：TP=`3781` TN=`992` FP=`1533` FN=`1888`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4973` F1=`0.5585` BalAcc=`0.5123`
- `stieger_only`：Acc=`0.5865` F1=`0.6937` BalAcc=`0.5304`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5907`
- Val F1（最优 checkpoint 时，附报）：`0.7286`
- Val loss（最优时）：`0.6458`

**Test（overall）**
- Accuracy：`0.6405`
- Recall：`0.7818`
- Specificity：`0.2983`
- Precision：`0.7296`
- F1：`0.7548`
- Balanced Acc：`0.5400`
- 混淆矩阵：TP=`4608` TN=`726` FP=`1708` FN=`1286`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4132` F1=`0.3288` BalAcc=`0.5446`
- `stieger_only`：Acc=`0.6451` F1=`0.7599` BalAcc=`0.5396`

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
  "arm": "B1",
  "task_sampler": "balanced_invfreq",
  "task_weight_mode": "none",
  "task_w0": 1.0,
  "task_w1": 1.0,
  "task_early_stop": "balanced_accuracy",
  "task_only": true
}
```

- 结束：`2026-08-01T16:54:58`
