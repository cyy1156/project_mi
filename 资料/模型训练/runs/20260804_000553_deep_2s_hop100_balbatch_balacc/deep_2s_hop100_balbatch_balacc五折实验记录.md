# 被试独立五折实验记录（20260804_000553 / deep_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T00:05:53`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`deep`（原结构）
- 结构：Deep4Net（braindecode 默认；n_times=500）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\deep_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_000553`

---
## 最终结论（Task only）

- Val BalAcc：`0.5009 ± 0.0018`
- Test BalAcc：`0.5055 ± 0.0110`
- Test Spec：`0.9827 ± 0.0347`
- Test Rec：`0.0284 ± 0.0567`
- Test F1：`0.0481 ± 0.0958`
- Test Acc：`0.3280 ± 0.0314`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`0.9925`

**Test（overall）**
- Accuracy：`0.3197`
- Recall：`0.0004`
- Specificity：`1.0000`
- Precision：`1.0000`
- F1：`0.0007`
- Balanced Acc：`0.5002`
- 混淆矩阵：TP=`2` TN=`2632` FP=`0` FN=`5605`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`1.1481`

**Test（overall）**
- Accuracy：`0.3163`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2614` FP=`0` FN=`5649`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5045`
- Val F1（最优 checkpoint 时，附报）：`0.1580`
- Val loss（最优时）：`0.7704`

**Test（overall）**
- Accuracy：`0.3892`
- Recall：`0.1418`
- Specificity：`0.9134`
- Precision：`0.7761`
- F1：`0.2397`
- Balanced Acc：`0.5276`
- 混淆矩阵：TP=`780` TN=`2372` FP=`225` FN=`4722`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`1.3794`

**Test（overall）**
- Accuracy：`0.3150`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`2405` FP=`0` FN=`5229`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc）：`0.5002`
- Val F1（最优 checkpoint 时，附报）：`0.0007`
- Val loss（最优时）：`1.6439`

**Test（overall）**
- Accuracy：`0.2996`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`1042` FP=`0` FN=`2436`

### 共用超参
```json
{
  "data_tag": "bci2a_2s_hop100",
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
  "protocol": "2s-hop100ms-offline-native",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true
}
```

- 结束：`2026-08-04T00:11:49`
