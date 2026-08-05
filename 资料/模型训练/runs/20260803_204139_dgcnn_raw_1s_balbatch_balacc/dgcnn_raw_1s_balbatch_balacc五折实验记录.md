# 被试独立五折实验记录（20260803_204139 / dgcnn_raw_1s_balbatch_balacc）

- 开始：`2026-08-03T20:41:39`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`dgcnn_raw`（原结构）
- 结构：TemporalEncoder(D=64) + DGCNN(k=2)；1s 原始时域 (B,8,250)
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\dgcnn_raw_1s_balbatch_balacc\bci2a_1s\run_20260803_204139`

---
## 最终结论（Task only）

- Val BalAcc：`0.5727 ± 0.0293`
- Test BalAcc：`0.5300 ± 0.0067`
- Test Spec：`0.6664 ± 0.2220`
- Test Rec：`0.3936 ± 0.2275`
- Test F1：`0.4635 ± 0.2100`
- Test Acc：`0.4774 ± 0.0888`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`62`
- 验证最优轮次（best_epoch）：`44`
- Val 选模分数（Balanced Acc）：`0.6170`
- Val F1（最优 checkpoint 时，附报）：`0.7098`
- Val loss（最优时）：`0.6551`

**Test（overall）**
- Accuracy：`0.4498`
- Recall：`0.2963`
- Specificity：`0.7730`
- Precision：`0.7331`
- F1：`0.4220`
- Balanced Acc：`0.5346`
- 混淆矩阵：TP=`6012` TN=`7454` FP=`2189` FN=`14280`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val 选模分数（Balanced Acc）：`0.5892`
- Val F1（最优 checkpoint 时，附报）：`0.6710`
- Val loss（最优时）：`0.6719`

**Test（overall）**
- Accuracy：`0.4617`
- Recall：`0.3415`
- Specificity：`0.7176`
- Precision：`0.7202`
- F1：`0.4633`
- Balanced Acc：`0.5296`
- 混淆矩阵：TP=`6982` TN=`6892` FP=`2712` FN=`13462`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`63`
- 验证最优轮次（best_epoch）：`45`
- Val 选模分数（Balanced Acc）：`0.5486`
- Val F1（最优 checkpoint 时，附报）：`0.6963`
- Val loss（最优时）：`0.6754`

**Test（overall）**
- Accuracy：`0.5896`
- Recall：`0.7065`
- Specificity：`0.3455`
- Precision：`0.6926`
- F1：`0.6995`
- Balanced Acc：`0.5260`
- 混淆矩阵：TP=`14068` TN=`3295` FP=`6243` FN=`5844`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`102`
- 验证最优轮次（best_epoch）：`84`
- Val 选模分数（Balanced Acc）：`0.5739`
- Val F1（最优 checkpoint 时，附报）：`0.6427`
- Val loss（最优时）：`0.7070`

**Test（overall）**
- Accuracy：`0.5514`
- Recall：`0.5722`
- Specificity：`0.5070`
- Precision：`0.7129`
- F1：`0.6348`
- Balanced Acc：`0.5396`
- 混淆矩阵：TP=`10828` TN=`4483` FP=`4360` FN=`8096`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`39`
- 验证最优轮次（best_epoch）：`21`
- Val 选模分数（Balanced Acc）：`0.5344`
- Val F1（最优 checkpoint 时，附报）：`0.6044`
- Val loss（最优时）：`0.7081`

**Test（overall）**
- Accuracy：`0.3347`
- Recall：`0.0517`
- Specificity：`0.9890`
- Precision：`0.9157`
- F1：`0.0979`
- Balanced Acc：`0.5204`
- 混淆矩阵：TP=`456` TN=`3770` FP=`42` FN=`8360`

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

- 结束：`2026-08-03T21:55:54`
