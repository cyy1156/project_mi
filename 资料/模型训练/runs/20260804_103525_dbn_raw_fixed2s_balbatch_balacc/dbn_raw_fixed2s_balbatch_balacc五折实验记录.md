# 被试独立五折实验记录（20260804_103525 / dbn_raw_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:35:25`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`dbn_raw`（原结构）
- 结构：TemporalEncoder(D=64) + DBN；fixed2s 原始时域 (B,8,500)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dbn_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103525`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5899 ± 0.0320`
- Test BalAcc：`0.5351 ± 0.0595`
- Test Spec：`0.8332 ± 0.1572`
- Test Rec：`0.2371 ± 0.2270`
- Test F1：`0.3037 ± 0.2582`
- Test Acc：`0.4272 ± 0.1155`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`47`
- 验证最优轮次（best_epoch）：`29`
- Val 选模分数（Balanced Acc）：`0.6002`
- Val F1（最优 checkpoint 时，附报）：`0.3803`
- Val loss（最优时）：`0.7098`

**Test（overall）**
- Accuracy：`0.5455`
- Recall：`0.3933`
- Specificity：`0.8605`
- Precision：`0.8537`
- F1：`0.5385`
- Balanced Acc：`0.6269`
- 混淆矩阵：TP=`105` TN=`111` FP=`18` FN=`162`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.5890`
- Val F1（最优 checkpoint 时，附报）：`0.4667`
- Val loss（最优时）：`0.6907`

**Test（overall）**
- Accuracy：`0.3392`
- Recall：`0.0372`
- Specificity：`0.9690`
- Precision：`0.7143`
- F1：`0.0707`
- Balanced Acc：`0.5031`
- 混淆矩阵：TP=`10` TN=`125` FP=`4` FN=`259`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5495`
- Val F1（最优 checkpoint 时，附报）：`0.2297`
- Val loss（最优时）：`0.6965`

**Test（overall）**
- Accuracy：`0.3590`
- Recall：`0.1565`
- Specificity：`0.7734`
- Precision：`0.5857`
- F1：`0.2470`
- Balanced Acc：`0.4650`
- 混淆矩阵：TP=`41` TN=`99` FP=`29` FN=`221`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val 选模分数（Balanced Acc）：`0.5671`
- Val F1（最优 checkpoint 时，附报）：`0.3636`
- Val loss（最优时）：`0.7544`

**Test（overall）**
- Accuracy：`0.5870`
- Recall：`0.5984`
- Specificity：`0.5630`
- Precision：`0.7413`
- F1：`0.6622`
- Balanced Acc：`0.5807`
- 混淆矩阵：TP=`149` TN=`67` FP=`52` FN=`100`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`58`
- 验证最优轮次（best_epoch）：`40`
- Val 选模分数（Balanced Acc）：`0.6435`
- Val F1（最优 checkpoint 时，附报）：`0.6798`
- Val loss（最优时）：`0.6596`

**Test（overall）**
- Accuracy：`0.3054`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`51` FP=`0` FN=`116`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4475 ± 0.0435`
- Val F1-macro：`0.4022 ± 0.0759`
- Test BalAcc：`0.3751 ± 0.0471`
- Test F1-macro：`0.2798 ± 0.1081`
- Test Acc：`0.3680 ± 0.0557`
- Test Precision-macro：`0.3037 ± 0.1416`
- Test Recall-macro：`0.3751 ± 0.0471`
- Test Recall idle/left/right：`0.5899±0.3299` / `0.3495±0.3226` / `0.1860±0.3024`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`67`
- 验证最优轮次（best_epoch）：`49`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5292`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5168`
- Val loss（最优时）：`1.0020`

**Test（overall）**
- Accuracy：`0.4571`
- Balanced Acc：`0.4521`
- F1-macro：`0.4300`
- Precision-macro：`0.4912`
- Recall-macro：`0.4521`
- Recall idle/left/right：`0.2946` / `0.2824` / `0.7794`
- Precision idle/left/right：`0.5352` / `0.5211` / `0.4173`
- F1 idle/left/right：`0.3800` / `0.3663` / `0.5436`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     38     13     78
  true1     24     37     70
  true2      9     21    106
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4025`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3008`
- Val loss（最优时）：`1.0990`

**Test（overall）**
- Accuracy：`0.3241`
- Balanced Acc：`0.3330`
- F1-macro：`0.1756`
- Precision-macro：`0.2503`
- Recall-macro：`0.3330`
- Recall idle/left/right：`0.9767` / `0.0222` / `0.0000`
- Precision idle/left/right：`0.3223` / `0.4286` / `0.0000`
- F1 idle/left/right：`0.4846` / `0.0423` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    126      3      0
  true1    132      3      0
  true2    133      1      0
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4278`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3384`
- Val loss（最优时）：`1.1003`

**Test（overall）**
- Accuracy：`0.3487`
- Balanced Acc：`0.3500`
- F1-macro：`0.2609`
- Precision-macro：`0.2397`
- Recall-macro：`0.3500`
- Recall idle/left/right：`0.2578` / `0.7923` / `0.0000`
- Precision idle/left/right：`0.3793` / `0.3399` / `0.0000`
- F1 idle/left/right：`0.3070` / `0.4758` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     33     95      0
  true1     27    103      0
  true2     27    105      0
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`53`
- 验证最优轮次（best_epoch）：`35`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4495`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4278`
- Val loss（最优时）：`1.0819`

**Test（overall）**
- Accuracy：`0.4049`
- Balanced Acc：`0.4071`
- F1-macro：`0.3767`
- Precision-macro：`0.4354`
- Recall-macro：`0.4071`
- Recall idle/left/right：`0.4202` / `0.6504` / `0.1508`
- Precision idle/left/right：`0.4098` / `0.3828` / `0.5135`
- F1 idle/left/right：`0.4149` / `0.4819` / `0.2331`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     50     59     10
  true1     35     80      8
  true2     37     70     19
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4284`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4274`
- Val loss（最优时）：`1.0741`

**Test（overall）**
- Accuracy：`0.3054`
- Balanced Acc：`0.3333`
- F1-macro：`0.1560`
- Precision-macro：`0.1018`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- Precision idle/left/right：`0.3054` / `0.0000` / `0.0000`
- F1 idle/left/right：`0.4679` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     51      0      0
  true1     53      0      0
  true2     63      0      0
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

- 结束：`2026-08-04T10:36:40`
