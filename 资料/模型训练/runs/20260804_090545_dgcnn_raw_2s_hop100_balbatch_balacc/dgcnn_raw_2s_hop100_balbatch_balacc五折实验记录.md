# 被试独立五折实验记录（20260804_090545 / dgcnn_raw_2s_hop100_balbatch_balacc）

- 开始：`2026-08-04T09:05:45`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s_hop100`（prefix=`bci2a`，**单库不合并**）
- protocol：`2s-hop100ms-offline-native` | **no_rap=True**
- 读数口径：`Tw=2s hop=100ms fu=10Hz no_paper_align`
- model：`dgcnn_raw`（原结构）
- 结构：TemporalEncoder(D=64) + DGCNN(k=2)；2s/hop100 原始时域 (B,8,500)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s_hop100', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '2s-hop100ms-offline-native', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_2s_hop100\dgcnn_raw_2s_hop100_balbatch_balacc\bci2a_2s_hop100\run_20260804_090545`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5842 ± 0.0735`
- Test BalAcc：`0.5266 ± 0.0217`
- Test Spec：`0.7394 ± 0.2388`
- Test Rec：`0.3138 ± 0.2762`
- Test F1：`0.3714 ± 0.2537`
- Test Acc：`0.4459 ± 0.1180`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val 选模分数（Balanced Acc）：`0.7069`
- Val F1（最优 checkpoint 时，附报）：`0.7732`
- Val loss（最优时）：`0.6431`

**Test（overall）**
- Accuracy：`0.4189`
- Recall：`0.2620`
- Specificity：`0.7530`
- Precision：`0.6933`
- F1：`0.3803`
- Balanced Acc：`0.5075`
- 混淆矩阵：TP=`1469` TN=`1982` FP=`650` FN=`4138`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`53`
- 验证最优轮次（best_epoch）：`35`
- Val 选模分数（Balanced Acc）：`0.5573`
- Val F1（最优 checkpoint 时，附报）：`0.5763`
- Val loss（最优时）：`0.7336`

**Test（overall）**
- Accuracy：`0.3828`
- Recall：`0.1368`
- Specificity：`0.9143`
- Precision：`0.7753`
- F1：`0.2326`
- Balanced Acc：`0.5256`
- 混淆矩阵：TP=`773` TN=`2390` FP=`224` FN=`4876`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc）：`0.6226`
- Val F1（最优 checkpoint 时，附报）：`0.6201`
- Val loss（最优时）：`0.6844`

**Test（overall）**
- Accuracy：`0.4757`
- Recall：`0.3597`
- Specificity：`0.7216`
- Precision：`0.7324`
- F1：`0.4824`
- Balanced Acc：`0.5406`
- 混淆矩阵：TP=`1979` TN=`1874` FP=`723` FN=`3523`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5367`
- Val F1（最优 checkpoint 时，附报）：`0.7736`
- Val loss（最优时）：`0.6872`

**Test（overall）**
- Accuracy：`0.6523`
- Recall：`0.8107`
- Specificity：`0.3081`
- Precision：`0.7181`
- F1：`0.7616`
- Balanced Acc：`0.5594`
- 混淆矩阵：TP=`4239` TN=`741` FP=`1664` FN=`990`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.4975`
- Val F1（最优 checkpoint 时，附报）：`0.1410`
- Val loss（最优时）：`0.7493`

**Test（overall）**
- Accuracy：`0.2996`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`1042` FP=`0` FN=`2436`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4399 ± 0.0619`
- Val F1-macro：`0.4315 ± 0.0629`
- Test BalAcc：`0.3709 ± 0.0169`
- Test F1-macro：`0.3133 ± 0.0640`
- Test Acc：`0.3611 ± 0.0245`
- Test Precision-macro：`0.4482 ± 0.0946`
- Test Recall-macro：`0.3709 ± 0.0169`
- Test Recall idle/left/right：`0.6501±0.2463` / `0.3054±0.1701` / `0.1572±0.1187`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5547`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5454`
- Val loss（最优时）：`0.9925`

**Test（overall）**
- Accuracy：`0.3492`
- Balanced Acc：`0.3577`
- F1-macro：`0.3080`
- Precision-macro：`0.3883`
- Recall-macro：`0.3577`
- Recall idle/left/right：`0.7036` / `0.2832` / `0.0861`
- Precision idle/left/right：`0.3139` / `0.4524` / `0.3987`
- F1 idle/left/right：`0.4341` / `0.3483` / `0.1417`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1852    517    263
  true1   1864    779    108
  true2   2184    426    246
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4527`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4530`
- Val loss（最优时）：`1.0617`

**Test（overall）**
- Accuracy：`0.3970`
- Balanced Acc：`0.4018`
- F1-macro：`0.3885`
- Precision-macro：`0.4295`
- Recall-macro：`0.4018`
- Recall idle/left/right：`0.5968` / `0.3531` / `0.2555`
- Precision idle/left/right：`0.3434` / `0.4313` / `0.5139`
- F1 idle/left/right：`0.4359` / `0.3883` / `0.3413`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1560    663    391
  true1   1545   1001    289
  true2   1438    657    719
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3827`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3750`
- Val loss（最优时）：`1.0973`

**Test（overall）**
- Accuracy：`0.3531`
- Balanced Acc：`0.3606`
- F1-macro：`0.2972`
- Precision-macro：`0.3973`
- Recall-macro：`0.3606`
- Recall idle/left/right：`0.7151` / `0.3165` / `0.0501`
- Precision idle/left/right：`0.3410` / `0.3650` / `0.4860`
- F1 idle/left/right：`0.4618` / `0.3390` / `0.0909`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1857    672     68
  true1   1787    864     79
  true2   1802    831    139
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3998`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3876`
- Val loss（最优时）：`1.1001`

**Test（overall）**
- Accuracy：`0.3794`
- Balanced Acc：`0.3764`
- F1-macro：`0.3672`
- Precision-macro：`0.3908`
- Recall-macro：`0.3764`
- Recall idle/left/right：`0.2358` / `0.5528` / `0.3405`
- Precision idle/left/right：`0.4169` / `0.3548` / `0.4006`
- F1 idle/left/right：`0.3012` / `0.4322` / `0.3681`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    567   1264    574
  true1    381   1428    774
  true2    412   1333    901
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`92`
- 验证最优轮次（best_epoch）：`74`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4097`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3962`
- Val loss（最优时）：`1.2182`

**Test（overall）**
- Accuracy：`0.3266`
- Balanced Acc：`0.3581`
- F1-macro：`0.2054`
- Precision-macro：`0.6352`
- Recall-macro：`0.3581`
- Recall idle/left/right：`0.9990` / `0.0216` / `0.0537`
- Precision idle/left/right：`0.3116` / `0.9600` / `0.6339`
- F1 idle/left/right：`0.4750` / `0.0422` / `0.0990`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1041      1      0
  true1   1048     24     41
  true2   1252      0     71
```

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

- 结束：`2026-08-04T09:35:26`
