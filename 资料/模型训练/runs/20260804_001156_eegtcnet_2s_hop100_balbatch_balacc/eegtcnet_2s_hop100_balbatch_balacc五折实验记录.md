# 被试独立五折实验记录（20260804_001156 / eegtcnet_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T00:11:56`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`eegtcnet`（原结构）
- 结构：EEGTCNet（braindecode 默认）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\eegtcnet_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_001156`

---
## 最终结论（Task only）

- Val BalAcc：`0.5901 ± 0.0691`
- Test BalAcc：`0.5856 ± 0.0284`
- Test Spec：`0.6770 ± 0.1685`
- Test Rec：`0.4943 ± 0.1700`
- Test F1：`0.5807 ± 0.1385`
- Test Acc：`0.5494 ± 0.0724`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`110`
- 验证最优轮次（best_epoch）：`92`
- Val 选模分数（Balanced Acc）：`0.7167`
- Val F1（最优 checkpoint 时，附报）：`0.8098`
- Val loss（最优时）：`0.5372`

**Test（overall）**
- Accuracy：`0.5553`
- Recall：`0.5104`
- Specificity：`0.6508`
- Precision：`0.7569`
- F1：`0.6097`
- Balanced Acc：`0.5806`
- 混淆矩阵：TP=`2862` TN=`1713` FP=`919` FN=`2745`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5411`
- Val F1（最优 checkpoint 时，附报）：`0.5505`
- Val loss（最优时）：`0.6943`

**Test（overall）**
- Accuracy：`0.5601`
- Recall：`0.4826`
- Specificity：`0.7276`
- Precision：`0.7929`
- F1：`0.6000`
- Balanced Acc：`0.6051`
- 混淆矩阵：TP=`2726` TN=`1902` FP=`712` FN=`2923`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.5208`
- Val F1（最优 checkpoint 时，附报）：`0.6782`
- Val loss（最优时）：`0.6950`

**Test（overall）**
- Accuracy：`0.6545`
- Recall：`0.7130`
- Specificity：`0.5306`
- Precision：`0.7629`
- F1：`0.7371`
- Balanced Acc：`0.6218`
- 混淆矩阵：TP=`3923` TN=`1378` FP=`1219` FN=`1579`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc）：`0.5681`
- Val F1（最优 checkpoint 时，附报）：`0.6621`
- Val loss（最优时）：`0.7261`

**Test（overall）**
- Accuracy：`0.5503`
- Recall：`0.5718`
- Specificity：`0.5035`
- Precision：`0.7146`
- F1：`0.6353`
- Balanced Acc：`0.5377`
- 混淆矩阵：TP=`2990` TN=`1211` FP=`1194` FN=`2239`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`65`
- 验证最优轮次（best_epoch）：`47`
- Val 选模分数（Balanced Acc）：`0.6038`
- Val F1（最优 checkpoint 时，附报）：`0.7245`
- Val loss（最优时）：`0.6615`

**Test（overall）**
- Accuracy：`0.4270`
- Recall：`0.1938`
- Specificity：`0.9722`
- Precision：`0.9421`
- F1：`0.3214`
- Balanced Acc：`0.5830`
- 混淆矩阵：TP=`472` TN=`1013` FP=`29` FN=`1964`

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

- 结束：`2026-08-04T00:33:10`
