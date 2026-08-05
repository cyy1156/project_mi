# 被试独立五折实验记录（20260803_234154 / eegnet_2s_hop100_balbatch_balacc）

- 开始：`2026-08-03T23:41:54`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`eegnet`（原结构）
- 结构：EEGNet F1=8, D=2, F2=16（默认池化）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\eegnet_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260803_234154`

---
## 最终结论（Task only）

- Val BalAcc：`0.6190 ± 0.0567`
- Test BalAcc：`0.5920 ± 0.0239`
- Test Spec：`0.6784 ± 0.1683`
- Test Rec：`0.5056 ± 0.1666`
- Test F1：`0.5923 ± 0.1277`
- Test Acc：`0.5578 ± 0.0683`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.7132`
- Val F1（最优 checkpoint 时，附报）：`0.7374`
- Val loss（最优时）：`0.6014`

**Test（overall）**
- Accuracy：`0.5705`
- Recall：`0.5732`
- Specificity：`0.5646`
- Precision：`0.7372`
- F1：`0.6449`
- Balanced Acc：`0.5689`
- 混淆矩阵：TP=`3214` TN=`1486` FP=`1146` FN=`2393`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5690`
- Val F1（最优 checkpoint 时，附报）：`0.6661`
- Val loss（最优时）：`0.6706`

**Test（overall）**
- Accuracy：`0.5216`
- Recall：`0.4022`
- Specificity：`0.7796`
- Precision：`0.7978`
- F1：`0.5348`
- Balanced Acc：`0.5909`
- 混淆矩阵：TP=`2272` TN=`2038` FP=`576` FN=`3377`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val 选模分数（Balanced Acc）：`0.5573`
- Val F1（最优 checkpoint 时，附报）：`0.5212`
- Val loss（最优时）：`0.7475`

**Test（overall）**
- Accuracy：`0.6497`
- Recall：`0.6761`
- Specificity：`0.5938`
- Precision：`0.7791`
- F1：`0.7239`
- Balanced Acc：`0.6349`
- 混淆矩阵：TP=`3720` TN=`1542` FP=`1055` FN=`1782`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val 选模分数（Balanced Acc）：`0.6085`
- Val F1（最优 checkpoint 时，附报）：`0.6073`
- Val loss（最优时）：`0.7010`

**Test（overall）**
- Accuracy：`0.5980`
- Recall：`0.6449`
- Specificity：`0.4960`
- Precision：`0.7356`
- F1：`0.6873`
- Balanced Acc：`0.5705`
- 混淆矩阵：TP=`3372` TN=`1193` FP=`1212` FN=`1857`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`53`
- 验证最优轮次（best_epoch）：`35`
- Val 选模分数（Balanced Acc）：`0.6470`
- Val F1（最优 checkpoint 时，附报）：`0.7675`
- Val loss（最优时）：`0.6100`

**Test（overall）**
- Accuracy：`0.4491`
- Recall：`0.2315`
- Specificity：`0.9578`
- Precision：`0.9276`
- F1：`0.3706`
- Balanced Acc：`0.5947`
- 混淆矩阵：TP=`564` TN=`998` FP=`44` FN=`1872`

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

- 结束：`2026-08-03T23:59:20`
