# 被试独立五折实验记录（20260801_174726 / shallow_wce2_balbatch_balacc）

- 开始：`2026-08-01T17:47:26`
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
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balbatch_balacc\merged_2s\run_20260801_174726`

---
## 最终结论

### Task（静息/任务）— 臂 B2
- Val BalAcc（选模）：`0.5797 ± 0.0216`
- Test Spec：`0.5003 ± 0.1519`
- Test Rec：`0.6192 ± 0.1175`
- Test BalAcc：`0.5597 ± 0.0418`
- Test F1（附报）：`0.6736 ± 0.0685`
- Test Acc：`0.5826 ± 0.0520`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优（Balanced Acc）并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val 选模分数（Balanced Acc）：`0.6164`
- Val F1（最优 checkpoint 时，附报）：`0.6438`
- Val loss（最优时）：`0.7030`

**Test（overall）**
- Accuracy：`0.5303`
- Recall：`0.5188`
- Specificity：`0.5587`
- Precision：`0.7438`
- F1：`0.6113`
- Balanced Acc：`0.5388`
- 混淆矩阵：TP=`2619` TN=`1142` FP=`902` FN=`2429`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6313` F1=`0.7033` BalAcc=`0.6224`
- `stieger_only`：Acc=`0.5243` F1=`0.6057` BalAcc=`0.5339`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5604`
- Val F1（最优 checkpoint 时，附报）：`0.7730`
- Val loss（最优时）：`0.6292`

**Test（overall）**
- Accuracy：`0.5121`
- Recall：`0.4554`
- Specificity：`0.6567`
- Precision：`0.7718`
- F1：`0.5728`
- Balanced Acc：`0.5561`
- 混淆矩阵：TP=`1985` TN=`1123` FP=`587` FN=`2374`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5151` F1=`0.4683` BalAcc=`0.6231`
- `stieger_only`：Acc=`0.5119` F1=`0.5786` BalAcc=`0.5495`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.5599`
- Val F1（最优 checkpoint 时，附报）：`0.5547`
- Val loss（最优时）：`0.7376`

**Test（overall）**
- Accuracy：`0.6369`
- Recall：`0.6332`
- Specificity：`0.6477`
- Precision：`0.8408`
- F1：`0.7224`
- Balanced Acc：`0.6405`
- 混淆矩阵：TP=`3011` TN=`1048` FP=`570` FN=`1744`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7231` F1=`0.8051` BalAcc=`0.6560`
- `stieger_only`：Acc=`0.6313` F1=`0.7165` BalAcc=`0.6421`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5707`
- Val F1（最优 checkpoint 时，附报）：`0.6107`
- Val loss（最优时）：`0.7198`

**Test（overall）**
- Accuracy：`0.6006`
- Recall：`0.7271`
- Specificity：`0.3164`
- Precision：`0.7049`
- F1：`0.7158`
- Balanced Acc：`0.5218`
- 混淆矩阵：TP=`4122` TN=`799` FP=`1726` FN=`1547`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5245` F1=`0.6102` BalAcc=`0.5104`
- `stieger_only`：Acc=`0.6041` F1=`0.7201` BalAcc=`0.5220`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5912`
- Val F1（最优 checkpoint 时，附报）：`0.7120`
- Val loss（最优时）：`0.6551`

**Test（overall）**
- Accuracy：`0.6329`
- Recall：`0.7615`
- Specificity：`0.3217`
- Precision：`0.7311`
- F1：`0.7459`
- Balanced Acc：`0.5416`
- 混淆矩阵：TP=`4488` TN=`783` FP=`1651` FN=`1406`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3952` F1=`0.2937` BalAcc=`0.5317`
- `stieger_only`：Acc=`0.6378` F1=`0.7514` BalAcc=`0.5414`

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
  "task_w0": 1.05,
  "task_w1": 1.0,
  "task_early_stop": "balanced_accuracy",
  "task_only": true
}
```

- 结束：`2026-08-01T17:54:18`
