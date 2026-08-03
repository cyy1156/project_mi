# 被试独立五折实验记录（20260801_175902 / shallow_smote_balacc）

- 开始：`2026-08-01T17:59:02`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`shallow_smote_balacc`（新脚本 baseline_shallow_smote.py；不改 baseline_shallow.py）
- 臂：`S1` — 普通CE + train SMOTE(静息对齐任务) + BalAcc早停
- Task 重采样：仅 train 折 SMOTE（k=5，展平 8×T）；val/test 真实比例
- 早停：Balanced Acc；仅跑 Task（无 Three）
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1 仅附报）
- 结构：ShallowFBCSPNet（braindecode 默认结构 + shared drop_prob）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- weight_transfer：`False` | classifier：`native`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\shallow_smote_balacc\merged_2s\run_20260801_175902`

---
## 最终结论

### Task（静息/任务）— 臂 S1
- Val BalAcc（选模）：`0.5745 ± 0.0345`
- Test Spec：`0.3174 ± 0.1873`
- Test Rec：`0.8070 ± 0.1256`
- Test BalAcc：`0.5622 ± 0.0423`
- Test F1（附报）：`0.7708 ± 0.0516`
- Test Acc：`0.6654 ± 0.0467`

### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优（Balanced Acc）并保存权重的轮次。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val 选模分数（Balanced Acc）：`0.6327`
- Val F1（最优 checkpoint 时，附报）：`0.7064`
- Val loss（最优时）：`0.6999`

**Test（overall）**
- Accuracy：`0.6065`
- Recall：`0.6632`
- Specificity：`0.4662`
- Precision：`0.7542`
- F1：`0.7058`
- Balanced Acc：`0.5647`
- 混淆矩阵：TP=`3348` TN=`953` FP=`1091` FN=`1700`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7020` F1=`0.7855` BalAcc=`0.6448`
- `stieger_only`：Acc=`0.6008` F1=`0.7009` BalAcc=`0.5602`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val 选模分数（Balanced Acc）：`0.5335`
- Val F1（最优 checkpoint 时，附报）：`0.8006`
- Val loss（最优时）：`0.6981`

**Test（overall）**
- Accuracy：`0.6215`
- Recall：`0.6504`
- Specificity：`0.5480`
- Precision：`0.7858`
- F1：`0.7117`
- Balanced Acc：`0.5992`
- 混淆矩阵：TP=`2835` TN=`937` FP=`773` FN=`1524`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6005` F1=`0.6276` BalAcc=`0.6560`
- `stieger_only`：Acc=`0.6230` F1=`0.7164` BalAcc=`0.5933`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5694`
- Val F1（最优 checkpoint 时，附报）：`0.7658`
- Val loss（最优时）：`0.6792`

**Test（overall）**
- Accuracy：`0.7358`
- Recall：`0.8585`
- Specificity：`0.3752`
- Precision：`0.8015`
- F1：`0.8290`
- Balanced Acc：`0.6168`
- 混淆矩阵：TP=`4082` TN=`607` FP=`1011` FN=`673`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7128` F1=`0.8205` BalAcc=`0.5745`
- `stieger_only`：Acc=`0.7373` F1=`0.8296` BalAcc=`0.6221`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5884`
- Val F1（最优 checkpoint 时，附报）：`0.7336`
- Val loss（最优时）：`0.6617`

**Test（overall）**
- Accuracy：`0.6778`
- Recall：`0.9252`
- Specificity：`0.1224`
- Precision：`0.7030`
- F1：`0.7989`
- Balanced Acc：`0.5238`
- 混淆矩阵：TP=`5245` TN=`309` FP=`2216` FN=`424`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.6739` F1=`0.7692` BalAcc=`0.6033`
- `stieger_only`：Acc=`0.6780` F1=`0.8002` BalAcc=`0.5196`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5484`
- Val F1（最优 checkpoint 时，附报）：`0.8203`
- Val loss（最优时）：`0.5874`

**Test（overall）**
- Accuracy：`0.6855`
- Recall：`0.9376`
- Specificity：`0.0752`
- Precision：`0.7106`
- F1：`0.8084`
- Balanced Acc：`0.5064`
- 混淆矩阵：TP=`5526` TN=`183` FP=`2251` FN=`368`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4551` F1=`0.4417` BalAcc=`0.5473`
- `stieger_only`：Acc=`0.6902` F1=`0.8129` BalAcc=`0.5051`

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
  "arm": "S1",
  "task_resample": "smote",
  "smote_k_neighbors": 5,
  "task_weight_mode": "none",
  "task_w0": 1.0,
  "task_w1": 1.0,
  "task_early_stop": "balanced_accuracy",
  "task_only": true
}
```

- 结束：`2026-08-01T18:09:53`
