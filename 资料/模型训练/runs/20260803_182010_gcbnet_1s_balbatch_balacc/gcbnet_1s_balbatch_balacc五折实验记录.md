# 被试独立五折实验记录（20260803_182010 / gcbnet_1s_balbatch_balacc）

- 开始：`2026-08-03T18:20:10`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`gcbnet`（原结构）
- 结构：GCBNet(k=2, layers=[128]) + 1s bandpower
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\gcbnet_1s_balbatch_balacc\bci2a_1s\run_20260803_182010`

---
## 最终结论（Task only）

- Val BalAcc：`0.5569 ± 0.0470`
- Test BalAcc：`0.5345 ± 0.0138`
- Test Spec：`0.5021 ± 0.2564`
- Test Rec：`0.5668 ± 0.2367`
- Test F1：`0.5987 ± 0.1547`
- Test Acc：`0.5430 ± 0.0823`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.6494`
- Val F1（最优 checkpoint 时，附报）：`0.7633`
- Val loss（最优时）：`0.6559`

**Test（overall）**
- Accuracy：`0.5623`
- Recall：`0.5959`
- Specificity：`0.4914`
- Precision：`0.7115`
- F1：`0.6486`
- Balanced Acc：`0.5437`
- 混淆矩阵：TP=`12093` TN=`4739` FP=`4904` FN=`8199`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5270`
- Val F1（最优 checkpoint 时，附报）：`0.7493`
- Val loss（最优时）：`0.6692`

**Test（overall）**
- Accuracy：`0.4974`
- Recall：`0.4476`
- Specificity：`0.6036`
- Precision：`0.7062`
- F1：`0.5479`
- Balanced Acc：`0.5256`
- 混淆矩阵：TP=`9150` TN=`5797` FP=`3807` FN=`11294`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5381`
- Val F1（最优 checkpoint 时，附报）：`0.7970`
- Val loss（最优时）：`0.6489`

**Test（overall）**
- Accuracy：`0.6648`
- Recall：`0.9198`
- Specificity：`0.1324`
- Precision：`0.6888`
- F1：`0.7877`
- Balanced Acc：`0.5261`
- 混淆矩阵：TP=`18315` TN=`1263` FP=`8275` FN=`1597`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`124`
- 验证最优轮次（best_epoch）：`106`
- Val 选模分数（Balanced Acc）：`0.5463`
- Val F1（最优 checkpoint 时，附报）：`0.6914`
- Val loss（最优时）：`0.7479`

**Test（overall）**
- Accuracy：`0.5728`
- Recall：`0.6655`
- Specificity：`0.3744`
- Precision：`0.6948`
- F1：`0.6798`
- Balanced Acc：`0.5199`
- 混淆矩阵：TP=`12593` TN=`3311` FP=`5532` FN=`6331`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5237`
- Val F1（最优 checkpoint 时，附报）：`0.7444`
- Val loss（最优时）：`0.6691`

**Test（overall）**
- Accuracy：`0.4176`
- Recall：`0.2052`
- Specificity：`0.9087`
- Precision：`0.8387`
- F1：`0.3297`
- Balanced Acc：`0.5570`
- 混淆矩阵：TP=`1809` TN=`3464` FP=`348` FN=`7007`

### 共用超参
```json
{
  "data_tag": "bci2a_1s",
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
  "protocol": "1s-offline-native-arch",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 250,
  "no_rap": true
}
```

- 结束：`2026-08-03T18:59:00`
