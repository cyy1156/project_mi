# 被试独立五折实验记录（20260803_191800 / dbn_raw_1s_balbatch_balacc）

- 开始：`2026-08-03T19:18:00`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`dbn_raw`（原结构）
- 结构：TemporalEncoder(D=64) + DBN；1s 原始时域 (B,8,250)
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\dbn_raw_1s_balbatch_balacc\bci2a_1s\run_20260803_191800`

---
## 最终结论（Task only）

- Val BalAcc：`0.5562 ± 0.0281`
- Test BalAcc：`0.5201 ± 0.0118`
- Test Spec：`0.7389 ± 0.2800`
- Test Rec：`0.3013 ± 0.2946`
- Test F1：`0.3432 ± 0.2743`
- Test Acc：`0.4383 ± 0.1151`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`79`
- 验证最优轮次（best_epoch）：`61`
- Val 选模分数（Balanced Acc）：`0.6063`
- Val F1（最优 checkpoint 时，附报）：`0.6348`
- Val loss（最优时）：`0.7440`

**Test（overall）**
- Accuracy：`0.4180`
- Recall：`0.2086`
- Specificity：`0.8587`
- Precision：`0.7564`
- F1：`0.3270`
- Balanced Acc：`0.5336`
- 混淆矩阵：TP=`4233` TN=`8280` FP=`1363` FN=`16059`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5218`
- Val F1（最优 checkpoint 时，附报）：`0.3319`
- Val loss（最优时）：`0.7197`

**Test（overall）**
- Accuracy：`0.3347`
- Recall：`0.0340`
- Specificity：`0.9748`
- Precision：`0.7417`
- F1：`0.0650`
- Balanced Acc：`0.5044`
- 混淆矩阵：TP=`695` TN=`9362` FP=`242` FN=`19749`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`103`
- 验证最优轮次（best_epoch）：`85`
- Val 选模分数（Balanced Acc）：`0.5611`
- Val F1（最优 checkpoint 时，附报）：`0.5862`
- Val loss（最优时）：`0.8318`

**Test（overall）**
- Accuracy：`0.4977`
- Recall：`0.4420`
- Specificity：`0.6139`
- Precision：`0.7050`
- F1：`0.5434`
- Balanced Acc：`0.5280`
- 混淆矩阵：TP=`8802` TN=`5855` FP=`3683` FN=`11110`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5422`
- Val F1（最优 checkpoint 时，附报）：`0.7595`
- Val loss（最优时）：`0.6407`

**Test（overall）**
- Accuracy：`0.6278`
- Recall：`0.8046`
- Specificity：`0.2496`
- Precision：`0.6965`
- F1：`0.7466`
- Balanced Acc：`0.5271`
- 混淆矩阵：TP=`15226` TN=`2207` FP=`6636` FN=`3698`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5498`
- Val F1（最优 checkpoint 时，附报）：`0.5412`
- Val loss（最优时）：`0.7146`

**Test（overall）**
- Accuracy：`0.3132`
- Recall：`0.0172`
- Specificity：`0.9976`
- Precision：`0.9441`
- F1：`0.0339`
- Balanced Acc：`0.5074`
- 混淆矩阵：TP=`152` TN=`3803` FP=`9` FN=`8664`

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

- 结束：`2026-08-03T19:59:10`
