# 被试独立五折实验记录（20260803_091234 / eegnet_1s_balbatch_balacc）

- 开始：`2026-08-03T09:12:34`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`eegnet`（原结构）
- 结构：EEGNet F1=8, D=2, F2=16（默认池化）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\eegnet_1s_balbatch_balacc\bci2a_1s\run_20260803_091234`

---
## 最终结论（Task only）

- Val BalAcc：`0.5938 ± 0.0487`
- Test BalAcc：`0.5562 ± 0.0272`
- Test Spec：`0.7008 ± 0.1747`
- Test Rec：`0.4115 ± 0.2218`
- Test F1：`0.4911 ± 0.2003`
- Test Acc：`0.5009 ± 0.1002`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.6901`
- Val F1（最优 checkpoint 时，附报）：`0.7612`
- Val loss（最优时）：`0.6191`

**Test（overall）**
- Accuracy：`0.4777`
- Recall：`0.3569`
- Specificity：`0.7318`
- Precision：`0.7369`
- F1：`0.4809`
- Balanced Acc：`0.5444`
- 混淆矩阵：TP=`7242` TN=`7057` FP=`2586` FN=`13050`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc）：`0.5709`
- Val F1（最优 checkpoint 时，附报）：`0.5299`
- Val loss（最优时）：`0.7211`

**Test（overall）**
- Accuracy：`0.4427`
- Recall：`0.2934`
- Specificity：`0.7604`
- Precision：`0.7227`
- F1：`0.4174`
- Balanced Acc：`0.5269`
- 混淆矩阵：TP=`5998` TN=`7303` FP=`2301` FN=`14446`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5560`
- Val F1（最优 checkpoint 时，附报）：`0.6062`
- Val loss（最优时）：`0.6832`

**Test（overall）**
- Accuracy：`0.5939`
- Recall：`0.6296`
- Specificity：`0.5193`
- Precision：`0.7322`
- F1：`0.6771`
- Balanced Acc：`0.5745`
- 混淆矩阵：TP=`12537` TN=`4953` FP=`4585` FN=`7375`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5774`
- Val F1（最优 checkpoint 时，附报）：`0.5414`
- Val loss（最优时）：`0.6969`

**Test（overall）**
- Accuracy：`0.6323`
- Recall：`0.6892`
- Specificity：`0.5107`
- Precision：`0.7509`
- F1：`0.7187`
- Balanced Acc：`0.5999`
- 混淆矩阵：TP=`13042` TN=`4516` FP=`4327` FN=`5882`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val 选模分数（Balanced Acc）：`0.5744`
- Val F1（最优 checkpoint 时，附报）：`0.6698`
- Val loss（最优时）：`0.6700`

**Test（overall）**
- Accuracy：`0.3582`
- Recall：`0.0885`
- Specificity：`0.9819`
- Precision：`0.9187`
- F1：`0.1614`
- Balanced Acc：`0.5352`
- 混淆矩阵：TP=`780` TN=`3743` FP=`69` FN=`8036`

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

- 结束：`2026-08-03T10:23:53`
