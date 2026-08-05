# 被试独立五折实验记录（20260804_102628 / deep_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:26:28`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`deep`（原结构）
- 结构：Deep4Net（braindecode 默认；n_times=500；塌缩则改 compat 消融）
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\deep_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102628`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.6507 ± 0.0680`
- Test BalAcc：`0.5679 ± 0.0394`
- Test Spec：`0.5949 ± 0.2736`
- Test Rec：`0.5410 ± 0.2951`
- Test F1：`0.5666 ± 0.2433`
- Test Acc：`0.5542 ± 0.1215`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc）：`0.6974`
- Val F1（最优 checkpoint 时，附报）：`0.6776`
- Val loss（最优时）：`0.6410`

**Test（overall）**
- Accuracy：`0.6465`
- Recall：`0.6742`
- Specificity：`0.5891`
- Precision：`0.7725`
- F1：`0.7200`
- Balanced Acc：`0.6317`
- 混淆矩阵：TP=`180` TN=`76` FP=`53` FN=`87`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.7254`
- Val F1（最优 checkpoint 时，附报）：`0.7764`
- Val loss（最优时）：`0.6352`

**Test（overall）**
- Accuracy：`0.5980`
- Recall：`0.6022`
- Specificity：`0.5891`
- Precision：`0.7535`
- F1：`0.6694`
- Balanced Acc：`0.5957`
- 混淆矩阵：TP=`162` TN=`76` FP=`53` FN=`107`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5790`
- Val F1（最优 checkpoint 时，附报）：`0.8212`
- Val loss（最优时）：`0.6034`

**Test（overall）**
- Accuracy：`0.6846`
- Recall：`0.9504`
- Specificity：`0.1406`
- Precision：`0.6936`
- F1：`0.8019`
- Balanced Acc：`0.5455`
- 混淆矩阵：TP=`249` TN=`18` FP=`110` FN=`13`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`63`
- 验证最优轮次（best_epoch）：`45`
- Val 选模分数（Balanced Acc）：`0.6927`
- Val F1（最优 checkpoint 时，附报）：`0.7635`
- Val loss（最优时）：`0.6294`

**Test（overall）**
- Accuracy：`0.4946`
- Recall：`0.4177`
- Specificity：`0.6555`
- Precision：`0.7172`
- F1：`0.5279`
- Balanced Acc：`0.5366`
- 混淆矩阵：TP=`104` TN=`78` FP=`41` FN=`145`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5588`
- Val F1（最优 checkpoint 时，附报）：`0.2557`
- Val loss（最优时）：`0.7541`

**Test（overall）**
- Accuracy：`0.3473`
- Recall：`0.0603`
- Specificity：`1.0000`
- Precision：`1.0000`
- F1：`0.1138`
- Balanced Acc：`0.5302`
- 混淆矩阵：TP=`7` TN=`51` FP=`0` FN=`109`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4481 ± 0.0474`
- Val F1-macro：`0.3692 ± 0.0659`
- Test BalAcc：`0.3779 ± 0.0231`
- Test F1-macro：`0.3076 ± 0.0532`
- Test Acc：`0.3733 ± 0.0245`
- Test Precision-macro：`0.3310 ± 0.0894`
- Test Recall-macro：`0.3779 ± 0.0231`
- Test Recall idle/left/right：`0.4405±0.3624` / `0.3786±0.2602` / `0.3146±0.2321`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4579`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3615`
- Val loss（最优时）：`1.5900`

**Test（overall）**
- Accuracy：`0.3914`
- Balanced Acc：`0.3971`
- F1-macro：`0.3144`
- Precision-macro：`0.2621`
- Recall-macro：`0.3971`
- Recall idle/left/right：`0.5349` / `0.6565` / `0.0000`
- Precision idle/left/right：`0.4059` / `0.3805` / `0.0000`
- F1 idle/left/right：`0.4615` / `0.4818` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     69     60      0
  true1     45     86      0
  true2     56     80      0
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3634`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2665`
- Val loss（最优时）：`1.0849`

**Test（overall）**
- Accuracy：`0.3643`
- Balanced Acc：`0.3605`
- F1-macro：`0.3103`
- Precision-macro：`0.4045`
- Recall-macro：`0.3605`
- Recall idle/left/right：`0.0543` / `0.3481` / `0.6791`
- Precision idle/left/right：`0.4375` / `0.4476` / `0.3285`
- F1 idle/left/right：`0.0966` / `0.3917` / `0.4428`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      7     20    102
  true1      4     47     84
  true2      5     38     91
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4796`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3956`
- Val loss（最优时）：`1.0629`

**Test（overall）**
- Accuracy：`0.3590`
- Balanced Acc：`0.3570`
- F1-macro：`0.2820`
- Precision-macro：`0.2372`
- Recall-macro：`0.3570`
- Recall idle/left/right：`0.0000` / `0.6769` / `0.3939`
- Precision idle/left/right：`0.0000` / `0.3761` / `0.3355`
- F1 idle/left/right：`0.0000` / `0.4835` / `0.3624`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      0     67     61
  true1      0     88     42
  true2      1     79     52
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`81`
- 验证最优轮次（best_epoch）：`63`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5016`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4698`
- Val loss（最优时）：`1.0274`

**Test（overall）**
- Accuracy：`0.4103`
- Balanced Acc：`0.4136`
- F1-macro：`0.3973`
- Precision-macro：`0.4680`
- Recall-macro：`0.4136`
- Recall idle/left/right：`0.6723` / `0.2114` / `0.3571`
- Precision idle/left/right：`0.3390` / `0.4127` / `0.6522`
- F1 idle/left/right：`0.4507` / `0.2796` / `0.4615`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     80     24     15
  true1     88     26      9
  true2     68     13     45
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4382`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3527`
- Val loss（最优时）：`1.6748`

**Test（overall）**
- Accuracy：`0.3413`
- Balanced Acc：`0.3613`
- F1-macro：`0.2342`
- Precision-macro：`0.2831`
- Recall-macro：`0.3613`
- Recall idle/left/right：`0.9412` / `0.0000` / `0.1429`
- Precision idle/left/right：`0.3200` / `0.0000` / `0.5294`
- F1 idle/left/right：`0.4776` / `0.0000` / `0.2250`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     48      0      3
  true1     48      0      5
  true2     54      0      9
```

### 共用超参
```json
{
  "data_tag": "bci2a_2s",
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
  "protocol": "fixed-2s-cue2to4-bci2a",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 500,
  "no_rap": true
}
```

- 结束：`2026-08-04T10:27:35`
