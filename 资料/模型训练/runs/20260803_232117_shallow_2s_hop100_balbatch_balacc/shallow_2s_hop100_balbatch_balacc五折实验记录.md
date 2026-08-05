# 被试独立五折实验记录（20260803_232117 / shallow_2s_hop100_balbatch_balacc）

- 开始：`2026-08-03T23:21:17`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`shallow`（原结构）
- 结构：ShallowFBCSPNet（braindecode 默认）
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\shallow_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260803_232117`

---
## 最终结论（Task only）

- Val BalAcc：`0.6405 ± 0.0445`
- Test BalAcc：`0.6027 ± 0.0347`
- Test Spec：`0.5775 ± 0.1171`
- Test Rec：`0.6279 ± 0.0684`
- Test F1：`0.6873 ± 0.0323`
- Test Acc：`0.6113 ± 0.0247`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.7255`
- Val F1（最优 checkpoint 时，附报）：`0.8067`
- Val loss（最优时）：`0.5384`

**Test（overall）**
- Accuracy：`0.5673`
- Recall：`0.5762`
- Specificity：`0.5483`
- Precision：`0.7310`
- F1：`0.6445`
- Balanced Acc：`0.5622`
- 混淆矩阵：TP=`3231` TN=`1443` FP=`1189` FN=`2376`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`40`
- Val 选模分数（Balanced Acc）：`0.6351`
- Val F1（最优 checkpoint 时，附报）：`0.7691`
- Val loss（最优时）：`0.6985`

**Test（overall）**
- Accuracy：`0.6028`
- Recall：`0.5920`
- Specificity：`0.6262`
- Precision：`0.7739`
- F1：`0.6708`
- Balanced Acc：`0.6091`
- 混淆矩阵：TP=`3344` TN=`1637` FP=`977` FN=`2305`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5959`
- Val F1（最优 checkpoint 时，附报）：`0.7358`
- Val loss（最优时）：`0.6542`

**Test（overall）**
- Accuracy：`0.6302`
- Recall：`0.6627`
- Specificity：`0.5614`
- Precision：`0.7620`
- F1：`0.7089`
- Balanced Acc：`0.6120`
- 混淆矩阵：TP=`3646` TN=`1458` FP=`1139` FN=`1856`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.6202`
- Val F1（最优 checkpoint 时，附报）：`0.7296`
- Val loss（最优时）：`0.7123`

**Test（overall）**
- Accuracy：`0.6356`
- Recall：`0.7460`
- Specificity：`0.3954`
- Precision：`0.7285`
- F1：`0.7372`
- Balanced Acc：`0.5707`
- 混淆矩阵：TP=`3901` TN=`951` FP=`1454` FN=`1328`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.6257`
- Val F1（最优 checkpoint 时，附报）：`0.7699`
- Val loss（最优时）：`0.6488`

**Test（overall）**
- Accuracy：`0.6208`
- Recall：`0.5628`
- Specificity：`0.7562`
- Precision：`0.8437`
- F1：`0.6752`
- Balanced Acc：`0.6595`
- 混淆矩阵：TP=`1371` TN=`788` FP=`254` FN=`1065`

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

- 结束：`2026-08-03T23:29:41`
