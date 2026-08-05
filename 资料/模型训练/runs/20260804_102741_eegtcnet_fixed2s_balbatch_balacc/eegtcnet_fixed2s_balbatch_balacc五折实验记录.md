# 被试独立五折实验记录（20260804_102741 / eegtcnet_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:27:41`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`eegtcnet`（原结构）
- 结构：EEGTCNet（braindecode 默认）
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\eegtcnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102741`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5621 ± 0.0651`
- Test BalAcc：`0.5132 ± 0.0488`
- Test Spec：`0.2773 ± 0.2892`
- Test Rec：`0.7492 ± 0.3249`
- Test F1：`0.6658 ± 0.2394`
- Test Acc：`0.5988 ± 0.1350`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.8129`
- Val loss（最优时）：`0.6578`

**Test（overall）**
- Accuracy：`0.6742`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.6742`
- F1：`0.8054`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`267` TN=`0` FP=`129` FN=`0`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.4996`
- Val F1（最优 checkpoint 时，附报）：`0.0303`
- Val loss（最优时）：`0.7276`

**Test（overall）**
- Accuracy：`0.3367`
- Recall：`0.1152`
- Specificity：`0.7984`
- Precision：`0.5439`
- F1：`0.1902`
- Balanced Acc：`0.4568`
- 混淆矩阵：TP=`31` TN=`103` FP=`26` FN=`238`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`155`
- 验证最优轮次（best_epoch）：`137`
- Val 选模分数（Balanced Acc）：`0.6770`
- Val F1（最优 checkpoint 时，附报）：`0.8075`
- Val loss（最优时）：`0.6148`

**Test（overall）**
- Accuracy：`0.6590`
- Recall：`0.7977`
- Specificity：`0.3750`
- Precision：`0.7232`
- F1：`0.7586`
- Balanced Acc：`0.5864`
- 混淆矩阵：TP=`209` TN=`48` FP=`80` FN=`53`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.5769`
- Val F1（最优 checkpoint 时，附报）：`0.7958`
- Val loss（最优时）：`0.6407`

**Test（overall）**
- Accuracy：`0.6114`
- Recall：`0.8675`
- Specificity：`0.0756`
- Precision：`0.6626`
- F1：`0.7513`
- Balanced Acc：`0.4716`
- 混淆矩阵：TP=`216` TN=`9` FP=`110` FN=`33`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5570`
- Val F1（最优 checkpoint 时，附报）：`0.8109`
- Val loss（最优时）：`0.6381`

**Test（overall）**
- Accuracy：`0.7126`
- Recall：`0.9655`
- Specificity：`0.1373`
- Precision：`0.7179`
- F1：`0.8235`
- Balanced Acc：`0.5514`
- 混淆矩阵：TP=`112` TN=`7` FP=`44` FN=`4`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4157 ± 0.0459`
- Val F1-macro：`0.3466 ± 0.1001`
- Test BalAcc：`0.3481 ± 0.0172`
- Test F1-macro：`0.2756 ± 0.0597`
- Test Acc：`0.3463 ± 0.0198`
- Test Precision-macro：`0.3021 ± 0.0843`
- Test Recall-macro：`0.3481 ± 0.0172`
- Test Recall idle/left/right：`0.3442±0.3224` / `0.4589±0.3095` / `0.2412±0.2507`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`73`
- 验证最优轮次（best_epoch）：`55`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4341`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3427`
- Val loss（最优时）：`1.0936`

**Test（overall）**
- Accuracy：`0.3460`
- Balanced Acc：`0.3501`
- F1-macro：`0.2652`
- Precision-macro：`0.2299`
- Recall-macro：`0.3501`
- Recall idle/left/right：`0.3023` / `0.7481` / `0.0000`
- Precision idle/left/right：`0.3421` / `0.3475` / `0.0000`
- F1 idle/left/right：`0.3210` / `0.4746` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     39     90      0
  true1     33     98      0
  true2     42     94      0
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3333`
- Val F1-macro（最优 checkpoint 时，附报）：`0.1620`
- Val loss（最优时）：`1.1366`

**Test（overall）**
- Accuracy：`0.3166`
- Balanced Acc：`0.3255`
- F1-macro：`0.1644`
- Precision-macro：`0.1727`
- Recall-macro：`0.3255`
- Recall idle/left/right：`0.9690` / `0.0074` / `0.0000`
- Precision idle/left/right：`0.3181` / `0.2000` / `0.0000`
- F1 idle/left/right：`0.4789` / `0.0143` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    125      4      0
  true1    134      1      0
  true2    134      0      0
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4386`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4202`
- Val loss（最优时）：`1.0874`

**Test（overall）**
- Accuracy：`0.3385`
- Balanced Acc：`0.3358`
- F1-macro：`0.3001`
- Precision-macro：`0.3717`
- Recall-macro：`0.3358`
- Recall idle/left/right：`0.1641` / `0.1692` / `0.6742`
- Precision idle/left/right：`0.4667` / `0.3284` / `0.3201`
- F1 idle/left/right：`0.2428` / `0.2234` / `0.4341`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     21     14     93
  true1     12     22     96
  true2     12     31     89
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`91`
- 验证最优轮次（best_epoch）：`73`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4683`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4490`
- Val loss（最优时）：`1.0551`

**Test（overall）**
- Accuracy：`0.3533`
- Balanced Acc：`0.3531`
- F1-macro：`0.3289`
- Precision-macro：`0.3643`
- Recall-macro：`0.3531`
- Recall idle/left/right：`0.2269` / `0.6341` / `0.1984`
- Precision idle/left/right：`0.2755` / `0.3628` / `0.4545`
- F1 idle/left/right：`0.2488` / `0.4615` / `0.2762`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     27     76     16
  true1     31     78     14
  true2     40     61     25
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4044`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3591`
- Val loss（最优时）：`1.0902`

**Test（overall）**
- Accuracy：`0.3772`
- Balanced Acc：`0.3760`
- F1-macro：`0.3192`
- Precision-macro：`0.3719`
- Recall-macro：`0.3760`
- Recall idle/left/right：`0.0588` / `0.7358` / `0.3333`
- Precision idle/left/right：`0.3750` / `0.4021` / `0.3387`
- F1 idle/left/right：`0.1017` / `0.5200` / `0.3360`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      3     20     28
  true1      1     39     13
  true2      4     38     21
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

- 结束：`2026-08-04T10:29:49`
