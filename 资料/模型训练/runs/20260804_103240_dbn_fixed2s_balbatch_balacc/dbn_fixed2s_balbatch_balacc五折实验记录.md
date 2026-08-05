# 被试独立五折实验记录（20260804_103240 / dbn_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:32:40`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`dbn`（原结构）
- 结构：DBN + 2s μ/β log bandpower (N,8,2)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dbn_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103240`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.6002 ± 0.0254`
- Test BalAcc：`0.5116 ± 0.0196`
- Test Spec：`0.7084 ± 0.3826`
- Test Rec：`0.3148 ± 0.3716`
- Test F1：`0.3207 ± 0.3045`
- Test Acc：`0.4384 ± 0.1312`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`40`
- 验证最优轮次（best_epoch）：`22`
- Val 选模分数（Balanced Acc）：`0.6125`
- Val F1（最优 checkpoint 时，附报）：`0.8340`
- Val loss（最优时）：`0.6855`

**Test（overall）**
- Accuracy：`0.6742`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.6742`
- F1：`0.8054`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`267` TN=`0` FP=`129` FN=`0`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`63`
- 验证最优轮次（best_epoch）：`45`
- Val 选模分数（Balanced Acc）：`0.6134`
- Val F1（最优 checkpoint 时，附报）：`0.6867`
- Val loss（最优时）：`0.6891`

**Test（overall）**
- Accuracy：`0.3241`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`129` FP=`0` FN=`269`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.6259`
- Val F1（最优 checkpoint 时，附报）：`0.6870`
- Val loss（最优时）：`0.6895`

**Test（overall）**
- Accuracy：`0.4667`
- Recall：`0.3969`
- Specificity：`0.6094`
- Precision：`0.6753`
- F1：`0.5000`
- Balanced Acc：`0.5032`
- 混淆矩阵：TP=`104` TN=`78` FP=`50` FN=`158`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val 选模分数（Balanced Acc）：`0.5961`
- Val F1（最优 checkpoint 时，附报）：`0.6971`
- Val loss（最优时）：`0.6909`

**Test（overall）**
- Accuracy：`0.4158`
- Recall：`0.1687`
- Specificity：`0.9328`
- Precision：`0.8400`
- F1：`0.2809`
- Balanced Acc：`0.5507`
- 混淆矩阵：TP=`42` TN=`111` FP=`8` FN=`207`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`69`
- 验证最优轮次（best_epoch）：`51`
- Val 选模分数（Balanced Acc）：`0.5531`
- Val F1（最优 checkpoint 时，附报）：`0.8096`
- Val loss（最优时）：`0.6634`

**Test（overall）**
- Accuracy：`0.3114`
- Recall：`0.0086`
- Specificity：`1.0000`
- Precision：`1.0000`
- F1：`0.0171`
- Balanced Acc：`0.5043`
- 混淆矩阵：TP=`1` TN=`51` FP=`0` FN=`115`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.3530 ± 0.0243`
- Val F1-macro：`0.2134 ± 0.0592`
- Test BalAcc：`0.3357 ± 0.0048`
- Test F1-macro：`0.1853 ± 0.0419`
- Test Acc：`0.3308 ± 0.0142`
- Test Precision-macro：`0.1343 ± 0.0503`
- Test Recall-macro：`0.3357 ± 0.0048`
- Test Recall idle/left/right：`0.4688±0.4516` / `0.1385±0.2769` / `0.4000±0.4899`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3333`
- Val F1-macro（最优 checkpoint 时，附报）：`0.1598`
- Val loss（最优时）：`1.1019`

**Test（overall）**
- Accuracy：`0.3258`
- Balanced Acc：`0.3333`
- F1-macro：`0.1638`
- Precision-macro：`0.1086`
- Recall-macro：`0.3333`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0000`
- Precision idle/left/right：`0.3258` / `0.0000` / `0.0000`
- F1 idle/left/right：`0.4914` / `0.0000` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    129      0      0
  true1    131      0      0
  true2    136      0      0
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3874`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3023`
- Val loss（最优时）：`1.0991`

**Test（overall）**
- Accuracy：`0.3367`
- Balanced Acc：`0.3333`
- F1-macro：`0.1682`
- Precision-macro：`0.1125`
- Recall-macro：`0.3333`
- Recall idle/left/right：`0.0000` / `0.0000` / `1.0000`
- Precision idle/left/right：`0.0000` / `0.0000` / `0.3375`
- F1 idle/left/right：`0.0000` / `0.0000` / `0.5047`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      0      1    128
  true1      0      0    135
  true2      0      0    134
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3775`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2661`
- Val loss（最优时）：`1.1031`

**Test（overall）**
- Accuracy：`0.3436`
- Balanced Acc：`0.3454`
- F1-macro：`0.2685`
- Precision-macro：`0.2346`
- Recall-macro：`0.3454`
- Recall idle/left/right：`0.3438` / `0.6923` / `0.0000`
- Precision idle/left/right：`0.3729` / `0.3309` / `0.0000`
- F1 idle/left/right：`0.3577` / `0.4478` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     44     84      0
  true1     40     90      0
  true2     34     98      0
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3333`
- Val F1-macro（最优 checkpoint 时，附报）：`0.1738`
- Val loss（最优时）：`1.0985`

**Test（overall）**
- Accuracy：`0.3424`
- Balanced Acc：`0.3333`
- F1-macro：`0.1700`
- Precision-macro：`0.1141`
- Recall-macro：`0.3333`
- Recall idle/left/right：`0.0000` / `0.0000` / `1.0000`
- Precision idle/left/right：`0.0000` / `0.0000` / `0.3424`
- F1 idle/left/right：`0.0000` / `0.0000` / `0.5101`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      0      0    119
  true1      0      0    123
  true2      0      0    126
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3333`
- Val F1-macro（最优 checkpoint 时，附报）：`0.1647`
- Val loss（最优时）：`1.1018`

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

- 结束：`2026-08-04T10:33:08`
