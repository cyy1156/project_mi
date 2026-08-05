# 被试独立五折实验记录（20260804_102522 / shallow_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:25:22`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`shallow`（原结构）
- 结构：ShallowFBCSPNet（braindecode 默认）
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\shallow_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_102522`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.7230 ± 0.0632`
- Test BalAcc：`0.6539 ± 0.0788`
- Test Spec：`0.6096 ± 0.1587`
- Test Rec：`0.6982 ± 0.1087`
- Test F1：`0.7380 ± 0.0688`
- Test Acc：`0.6694 ± 0.0733`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`55`
- 验证最优轮次（best_epoch）：`37`
- Val 选模分数（Balanced Acc）：`0.8265`
- Val F1（最优 checkpoint 时，附报）：`0.8517`
- Val loss（最优时）：`0.4577`

**Test（overall）**
- Accuracy：`0.6338`
- Recall：`0.5955`
- Specificity：`0.7132`
- Precision：`0.8112`
- F1：`0.6868`
- Balanced Acc：`0.6543`
- 混淆矩阵：TP=`159` TN=`92` FP=`37` FN=`108`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.7523`
- Val F1（最优 checkpoint 时，附报）：`0.7489`
- Val loss（最优时）：`0.6130`

**Test（overall）**
- Accuracy：`0.7186`
- Recall：`0.6877`
- Specificity：`0.7829`
- Precision：`0.8685`
- F1：`0.7676`
- Balanced Acc：`0.7353`
- 混淆矩阵：TP=`185` TN=`101` FP=`28` FN=`84`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc）：`0.7107`
- Val F1（最优 checkpoint 时，附报）：`0.8078`
- Val loss（最优时）：`0.5445`

**Test（overall）**
- Accuracy：`0.7026`
- Recall：`0.8664`
- Specificity：`0.3672`
- Precision：`0.7370`
- F1：`0.7965`
- Balanced Acc：`0.6168`
- 混淆矩阵：TP=`227` TN=`47` FP=`81` FN=`35`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`71`
- 验证最优轮次（best_epoch）：`53`
- Val 选模分数（Balanced Acc）：`0.6846`
- Val F1（最优 checkpoint 时，附报）：`0.6822`
- Val loss（最优时）：`0.7779`

**Test（overall）**
- Accuracy：`0.5435`
- Recall：`0.5743`
- Specificity：`0.4790`
- Precision：`0.6976`
- F1：`0.6300`
- Balanced Acc：`0.5266`
- 混淆矩阵：TP=`143` TN=`57` FP=`62` FN=`106`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`73`
- 验证最优轮次（best_epoch）：`55`
- Val 选模分数（Balanced Acc）：`0.6408`
- Val F1（最优 checkpoint 时，附报）：`0.7875`
- Val loss（最优时）：`0.6264`

**Test（overall）**
- Accuracy：`0.7485`
- Recall：`0.7672`
- Specificity：`0.7059`
- Precision：`0.8558`
- F1：`0.8091`
- Balanced Acc：`0.7366`
- 混淆矩阵：TP=`89` TN=`36` FP=`15` FN=`27`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.5503 ± 0.0503`
- Val F1-macro：`0.5437 ± 0.0509`
- Test BalAcc：`0.5349 ± 0.0855`
- Test F1-macro：`0.5283 ± 0.0893`
- Test Acc：`0.5352 ± 0.0854`
- Test Precision-macro：`0.5510 ± 0.0805`
- Test Recall-macro：`0.5349 ± 0.0855`
- Test Recall idle/left/right：`0.5269±0.1710` / `0.5360±0.1333` / `0.5417±0.1543`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.6033`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5951`
- Val loss（最优时）：`0.9097`

**Test（overall）**
- Accuracy：`0.5101`
- Balanced Acc：`0.5063`
- F1-macro：`0.4932`
- Precision-macro：`0.5166`
- Recall-macro：`0.5063`
- Recall idle/left/right：`0.3953` / `0.3588` / `0.7647`
- Precision idle/left/right：`0.4811` / `0.5663` / `0.5024`
- F1 idle/left/right：`0.4340` / `0.4393` / `0.6064`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     51     19     59
  true1     40     47     44
  true2     15     17    104
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5419`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5325`
- Val loss（最优时）：`0.9719`

**Test（overall）**
- Accuracy：`0.5930`
- Balanced Acc：`0.5943`
- F1-macro：`0.5904`
- Precision-macro：`0.6006`
- Recall-macro：`0.5943`
- Recall idle/left/right：`0.7054` / `0.6000` / `0.4776`
- Precision idle/left/right：`0.5417` / `0.6328` / `0.6275`
- F1 idle/left/right：`0.6128` / `0.6160` / `0.5424`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     91     11     27
  true1     43     81     11
  true2     34     36     64
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`60`
- 验证最优轮次（best_epoch）：`42`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5758`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5754`
- Val loss（最优时）：`1.0302`

**Test（overall）**
- Accuracy：`0.4513`
- Balanced Acc：`0.4507`
- F1-macro：`0.4364`
- Precision-macro：`0.4901`
- Recall-macro：`0.4507`
- Recall idle/left/right：`0.2734` / `0.7000` / `0.3788`
- Precision idle/left/right：`0.5556` / `0.3939` / `0.5208`
- F1 idle/left/right：`0.3665` / `0.5042` / `0.4386`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     35     75     18
  true1     11     91     28
  true2     17     65     50
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5730`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5649`
- Val loss（最优时）：`0.9772`

**Test（overall）**
- Accuracy：`0.4511`
- Balanced Acc：`0.4526`
- F1-macro：`0.4504`
- Precision-macro：`0.4643`
- Recall-macro：`0.4526`
- Recall idle/left/right：`0.5546` / `0.3984` / `0.4048`
- Precision idle/left/right：`0.4049` / `0.4336` / `0.5543`
- F1 idle/left/right：`0.4681` / `0.4153` / `0.4679`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     66     38     15
  true1     48     49     26
  true2     49     26     51
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`62`
- 验证最优轮次（best_epoch）：`44`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4574`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4504`
- Val loss（最优时）：`1.1974`

**Test（overall）**
- Accuracy：`0.6707`
- Balanced Acc：`0.6704`
- F1-macro：`0.6708`
- Precision-macro：`0.6833`
- Recall-macro：`0.6704`
- Recall idle/left/right：`0.7059` / `0.6226` / `0.6825`
- Precision idle/left/right：`0.5455` / `0.7500` / `0.7544`
- F1 idle/left/right：`0.6154` / `0.6804` / `0.7167`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     36      8      7
  true1     13     33      7
  true2     17      3     43
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

- 结束：`2026-08-04T10:26:23`
