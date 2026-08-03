# 被试独立五折实验记录（20260801_173937 / shallow_wce2_balbatch_balacc）

- 开始：`2026-08-01T17:39:37`
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
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_wce2_balbatch_balacc\merged_2s\run_20260801_173937`

---
## 最终结论

### Task（静息/任务）— 臂 B2
- Val BalAcc（选模）：`0.5788 ± 0.0215`
- Test Spec：`0.5164 ± 0.1606`
- Test Rec：`0.5999 ± 0.1288`
- Test BalAcc：`0.5581 ± 0.0418`
- Test F1（附报）：`0.6606 ± 0.0756`
- Test Acc：`0.5736 ± 0.0567`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优（Balanced Acc）并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val 选模分数（Balanced Acc）：`0.6142`
- Val F1（最优 checkpoint 时，附报）：`0.6274`
- Val loss（最优时）：`0.7128`

**Test（overall）**
- Accuracy：`0.5173`
- Recall：`0.4945`
- Specificity：`0.5739`
- Precision：`0.7413`
- F1：`0.5932`
- Balanced Acc：`0.5342`
- 混淆矩阵：TP=`2496` TN=`1173` FP=`871` FN=`2552`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6212` F1=`0.6875` BalAcc=`0.6229`
- `stieger_only`：Acc=`0.5112` F1=`0.5875` BalAcc=`0.5289`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5592`
- Val F1（最优 checkpoint 时，附报）：`0.7593`
- Val loss（最优时）：`0.6419`

**Test（overall）**
- Accuracy：`0.5017`
- Recall：`0.4334`
- Specificity：`0.6760`
- Precision：`0.7732`
- F1：`0.5554`
- Balanced Acc：`0.5547`
- 混淆矩阵：TP=`1889` TN=`1156` FP=`554` FN=`2470`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4749` F1=`0.3871` BalAcc=`0.5994`
- `stieger_only`：Acc=`0.5036` F1=`0.5643` BalAcc=`0.5496`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.5579`
- Val F1（最优 checkpoint 时，附报）：`0.5310`
- Val loss（最优时）：`0.7467`

**Test（overall）**
- Accuracy：`0.6247`
- Recall：`0.6090`
- Specificity：`0.6706`
- Precision：`0.8446`
- F1：`0.7077`
- Balanced Acc：`0.6398`
- 混淆矩阵：TP=`2896` TN=`1085` FP=`533` FN=`1859`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7282` F1=`0.8051` BalAcc=`0.6718`
- `stieger_only`：Acc=`0.6179` F1=`0.7008` BalAcc=`0.6402`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5708`
- Val F1（最优 checkpoint 时，附报）：`0.5828`
- Val loss（最优时）：`0.7241`

**Test（overall）**
- Accuracy：`0.5791`
- Recall：`0.6624`
- Specificity：`0.3921`
- Precision：`0.7098`
- F1：`0.6853`
- Balanced Acc：`0.5272`
- 混淆矩阵：TP=`3755` TN=`990` FP=`1535` FN=`1914`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5245` F1=`0.6102` BalAcc=`0.5104`
- `stieger_only`：Acc=`0.5817` F1=`0.6885` BalAcc=`0.5279`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5917`
- Val F1（最优 checkpoint 时，附报）：`0.7287`
- Val loss（最优时）：`0.6523`

**Test（overall）**
- Accuracy：`0.6451`
- Recall：`0.8001`
- Specificity：`0.2695`
- Precision：`0.7262`
- F1：`0.7614`
- Balanced Acc：`0.5348`
- 混淆矩阵：TP=`4716` TN=`656` FP=`1778` FN=`1178`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5150` F1=`0.5475` BalAcc=`0.5740`
- `stieger_only`：Acc=`0.6477` F1=`0.7645` BalAcc=`0.5337`

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
  "task_w0": 1.1,
  "task_w1": 1.0,
  "task_early_stop": "balanced_accuracy",
  "task_only": true
}
```

- 结束：`2026-08-01T17:45:56`
