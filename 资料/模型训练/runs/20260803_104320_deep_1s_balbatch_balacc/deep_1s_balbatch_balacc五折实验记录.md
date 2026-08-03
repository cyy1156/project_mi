# 被试独立五折实验记录（20260803_104320 / deep_1s_balbatch_balacc）

- 开始：`2026-08-03T10:43:20`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`deep`（原结构）
- 结构：Deep4Net（braindecode 默认；250 点可能自动缩核）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\deep_1s_balbatch_balacc\bci2a_1s\run_20260803_104320`

---
## 最终结论（Task only）

- Val BalAcc：`0.5519 ± 0.0268`
- Test BalAcc：`0.5409 ± 0.0267`
- Test Spec：`0.6239 ± 0.3020`
- Test Rec：`0.4579 ± 0.3303`
- Test F1：`0.4932 ± 0.2327`
- Test Acc：`0.5082 ± 0.1307`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`49`
- 验证最优轮次（best_epoch）：`31`
- Val 选模分数（Balanced Acc）：`0.6031`
- Val F1（最优 checkpoint 时，附报）：`0.5083`
- Val loss（最优时）：`0.7759`

**Test（overall）**
- Accuracy：`0.3966`
- Recall：`0.1783`
- Specificity：`0.8561`
- Precision：`0.7227`
- F1：`0.2860`
- Balanced Acc：`0.5172`
- 混淆矩阵：TP=`3618` TN=`8255` FP=`1388` FN=`16674`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val 选模分数（Balanced Acc）：`0.5352`
- Val F1（最优 checkpoint 时，附报）：`0.2883`
- Val loss（最优时）：`0.9023`

**Test（overall）**
- Accuracy：`0.4009`
- Recall：`0.1959`
- Specificity：`0.8373`
- Precision：`0.7192`
- F1：`0.3079`
- Balanced Acc：`0.5166`
- 混淆矩阵：TP=`4004` TN=`8041` FP=`1563` FN=`16440`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`33`
- Val 选模分数（Balanced Acc）：`0.5546`
- Val F1（最优 checkpoint 时，附报）：`0.7780`
- Val loss（最优时）：`0.6467`

**Test（overall）**
- Accuracy：`0.6726`
- Recall：`0.9239`
- Specificity：`0.1478`
- Precision：`0.6936`
- F1：`0.7923`
- Balanced Acc：`0.5359`
- 混淆矩阵：TP=`18397` TN=`1410` FP=`8128` FN=`1515`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5325`
- Val F1（最优 checkpoint 时，附报）：`0.6623`
- Val loss（最优时）：`0.6959`

**Test（overall）**
- Accuracy：`0.6640`
- Recall：`0.7945`
- Specificity：`0.3846`
- Precision：`0.7342`
- F1：`0.7632`
- Balanced Acc：`0.5895`
- 混淆矩阵：TP=`15035` TN=`3401` FP=`5442` FN=`3889`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`54`
- 验证最优轮次（best_epoch）：`36`
- Val 选模分数（Balanced Acc）：`0.5343`
- Val F1（最优 checkpoint 时，附报）：`0.6420`
- Val loss（最优时）：`0.7296`

**Test（overall）**
- Accuracy：`0.4072`
- Recall：`0.1968`
- Specificity：`0.8938`
- Precision：`0.8107`
- F1：`0.3167`
- Balanced Acc：`0.5453`
- 混淆矩阵：TP=`1735` TN=`3407` FP=`405` FN=`7081`

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

- 结束：`2026-08-03T12:27:18`
