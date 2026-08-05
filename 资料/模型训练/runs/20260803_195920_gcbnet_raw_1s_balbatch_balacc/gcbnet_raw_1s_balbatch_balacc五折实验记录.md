# 被试独立五折实验记录（20260803_195920 / gcbnet_raw_1s_balbatch_balacc）

- 开始：`2026-08-03T19:59:20`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`gcbnet_raw`（原结构）
- 结构：TemporalEncoder(D=64) + GCBNet(k=2)；1s 原始时域 (B,8,250)
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\gcbnet_raw_1s_balbatch_balacc\bci2a_1s\run_20260803_195920`

---
## 最终结论（Task only）

- Val BalAcc：`0.5832 ± 0.0459`
- Test BalAcc：`0.5288 ± 0.0233`
- Test Spec：`0.5334 ± 0.1985`
- Test Rec：`0.5243 ± 0.1870`
- Test F1：`0.5783 ± 0.1422`
- Test Acc：`0.5245 ± 0.0710`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.6697`
- Val F1（最优 checkpoint 时，附报）：`0.7871`
- Val loss（最优时）：`0.6428`

**Test（overall）**
- Accuracy：`0.5673`
- Recall：`0.6675`
- Specificity：`0.3565`
- Precision：`0.6858`
- F1：`0.6765`
- Balanced Acc：`0.5120`
- 混淆矩阵：TP=`13545` TN=`3438` FP=`6205` FN=`6747`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.5808`
- Val F1（最优 checkpoint 时，附报）：`0.6066`
- Val loss（最优时）：`0.7024`

**Test（overall）**
- Accuracy：`0.4831`
- Recall：`0.4468`
- Specificity：`0.5605`
- Precision：`0.6839`
- F1：`0.5405`
- Balanced Acc：`0.5036`
- 混淆矩阵：TP=`9134` TN=`5383` FP=`4221` FN=`11310`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc）：`0.5444`
- Val F1（最优 checkpoint 时，附报）：`0.6927`
- Val loss（最优时）：`0.6734`

**Test（overall）**
- Accuracy：`0.5420`
- Recall：`0.5918`
- Specificity：`0.4381`
- Precision：`0.6874`
- F1：`0.6360`
- Balanced Acc：`0.5149`
- 混淆矩阵：TP=`11783` TN=`4179` FP=`5359` FN=`8129`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.5764`
- Val F1（最优 checkpoint 时，附报）：`0.7074`
- Val loss（最优时）：`0.6890`

**Test（overall）**
- Accuracy：`0.6179`
- Recall：`0.7173`
- Specificity：`0.4051`
- Precision：`0.7207`
- F1：`0.7190`
- Balanced Acc：`0.5612`
- 混淆矩阵：TP=`13574` TN=`3582` FP=`5261` FN=`5350`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val 选模分数（Balanced Acc）：`0.5446`
- Val F1（最优 checkpoint 时，附报）：`0.6925`
- Val loss（最优时）：`0.6901`

**Test（overall）**
- Accuracy：`0.4119`
- Recall：`0.1979`
- Specificity：`0.9069`
- Precision：`0.8310`
- F1：`0.3197`
- Balanced Acc：`0.5524`
- 混淆矩阵：TP=`1745` TN=`3457` FP=`355` FN=`7071`

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

- 结束：`2026-08-03T20:41:28`
