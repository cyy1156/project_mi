# 被试独立五折实验记录（20260804_004323 / deep_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T00:43:23`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`deep`（原结构）
- 结构：Deep4Net-compat（pool=1/1，对齐1s自动缩核；非满血stride3）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\deep_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_004323`

---
## 最终结论（Task only）

- Val BalAcc：`0.6194 ± 0.0514`
- Test BalAcc：`0.5702 ± 0.0263`
- Test Spec：`0.5389 ± 0.1550`
- Test Rec：`0.6016 ± 0.1255`
- Test F1：`0.6562 ± 0.0671`
- Test Acc：`0.5809 ± 0.0414`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`62`
- 验证最优轮次（best_epoch）：`44`
- Val 选模分数（Balanced Acc）：`0.7102`
- Val F1（最优 checkpoint 时，附报）：`0.8279`
- Val loss（最优时）：`0.5985`

**Test（overall）**
- Accuracy：`0.6419`
- Recall：`0.7731`
- Specificity：`0.3625`
- Precision：`0.7209`
- F1：`0.7461`
- Balanced Acc：`0.5678`
- 混淆矩阵：TP=`4335` TN=`954` FP=`1678` FN=`1272`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`99`
- 验证最优轮次（best_epoch）：`81`
- Val 选模分数（Balanced Acc）：`0.6199`
- Val F1（最优 checkpoint 时，附报）：`0.5650`
- Val loss（最优时）：`1.4757`

**Test（overall）**
- Accuracy：`0.5394`
- Recall：`0.4479`
- Specificity：`0.7372`
- Precision：`0.7864`
- F1：`0.5707`
- Balanced Acc：`0.5925`
- 混淆矩阵：TP=`2530` TN=`1927` FP=`687` FN=`3119`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`103`
- 验证最优轮次（best_epoch）：`85`
- Val 选模分数（Balanced Acc）：`0.6130`
- Val F1（最优 checkpoint 时，附报）：`0.7405`
- Val loss（最优时）：`1.0002`

**Test（overall）**
- Accuracy：`0.6008`
- Recall：`0.7186`
- Specificity：`0.3512`
- Precision：`0.7012`
- F1：`0.7098`
- Balanced Acc：`0.5349`
- 混淆矩阵：TP=`3954` TN=`912` FP=`1685` FN=`1548`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.5510`
- Val F1（最优 checkpoint 时，附报）：`0.5931`
- Val loss（最优时）：`0.8790`

**Test（overall）**
- Accuracy：`0.5300`
- Recall：`0.4963`
- Specificity：`0.6033`
- Precision：`0.7312`
- F1：`0.5913`
- Balanced Acc：`0.5498`
- 混淆矩阵：TP=`2595` TN=`1451` FP=`954` FN=`2634`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`40`
- Val 选模分数（Balanced Acc）：`0.6030`
- Val F1（最优 checkpoint 时，附报）：`0.7240`
- Val loss（最优时）：`0.8788`

**Test（overall）**
- Accuracy：`0.5926`
- Recall：`0.5722`
- Specificity：`0.6401`
- Precision：`0.7880`
- F1：`0.6630`
- Balanced Acc：`0.6062`
- 混淆矩阵：TP=`1394` TN=`667` FP=`375` FN=`1042`

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

- 结束：`2026-08-04T02:32:47`
