# 被试独立五折实验记录（20260804_103808 / dgcnn_raw_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:38:08`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`dgcnn_raw`（原结构）
- 结构：TemporalEncoder(D=64) + DGCNN(k=2)；fixed2s 原始时域 (B,8,500)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\dgcnn_raw_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103808`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5604 ± 0.0431`
- Test BalAcc：`0.5537 ± 0.0394`
- Test Spec：`0.7989 ± 0.1788`
- Test Rec：`0.3084 ± 0.2559`
- Test F1：`0.3806 ± 0.2495`
- Test Acc：`0.4659 ± 0.1151`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.6068`
- Val F1（最优 checkpoint 时，附报）：`0.4575`
- Val loss（最优时）：`0.9080`

**Test（overall）**
- Accuracy：`0.6717`
- Recall：`0.7640`
- Specificity：`0.4806`
- Precision：`0.7528`
- F1：`0.7584`
- Balanced Acc：`0.6223`
- 混淆矩阵：TP=`204` TN=`62` FP=`67` FN=`63`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5000`
- Val F1（最优 checkpoint 时，附报）：`0.0000`
- Val loss（最优时）：`6.1824`

**Test（overall）**
- Accuracy：`0.3241`
- Recall：`0.0000`
- Specificity：`1.0000`
- Precision：`0.0000`
- F1：`0.0000`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`0` TN=`129` FP=`0` FN=`269`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val 选模分数（Balanced Acc）：`0.6010`
- Val F1（最优 checkpoint 时，附报）：`0.5962`
- Val loss（最优时）：`0.6925`

**Test（overall）**
- Accuracy：`0.4615`
- Recall：`0.2977`
- Specificity：`0.7969`
- Precision：`0.7500`
- F1：`0.4262`
- Balanced Acc：`0.5473`
- 混淆矩阵：TP=`78` TN=`102` FP=`26` FN=`184`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc）：`0.5744`
- Val F1（最优 checkpoint 时，附报）：`0.4000`
- Val loss（最优时）：`0.6959`

**Test（overall）**
- Accuracy：`0.4049`
- Recall：`0.1526`
- Specificity：`0.9328`
- Precision：`0.8261`
- F1：`0.2576`
- Balanced Acc：`0.5427`
- 混淆矩阵：TP=`38` TN=`111` FP=`8` FN=`211`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5196`
- Val F1（最优 checkpoint 时，附报）：`0.4468`
- Val loss（最优时）：`0.6959`

**Test（overall）**
- Accuracy：`0.4671`
- Recall：`0.3276`
- Specificity：`0.7843`
- Precision：`0.7755`
- F1：`0.4606`
- Balanced Acc：`0.5559`
- 混淆矩阵：TP=`38` TN=`40` FP=`11` FN=`78`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.3944 ± 0.0556`
- Val F1-macro：`0.2908 ± 0.0952`
- Test BalAcc：`0.3685 ± 0.0214`
- Test F1-macro：`0.2866 ± 0.0754`
- Test Acc：`0.3619 ± 0.0323`
- Test Precision-macro：`0.3145 ± 0.1242`
- Test Recall-macro：`0.3685 ± 0.0214`
- Test Recall idle/left/right：`0.5804±0.3503` / `0.2466±0.2150` / `0.2785±0.2571`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3450`
- Val F1-macro（最优 checkpoint 时，附报）：`0.1850`
- Val loss（最优时）：`1.1032`

**Test（overall）**
- Accuracy：`0.3662`
- Balanced Acc：`0.3679`
- F1-macro：`0.2907`
- Precision-macro：`0.4154`
- Recall-macro：`0.3679`
- Recall idle/left/right：`0.7209` / `0.0076` / `0.3750`
- Precision idle/left/right：`0.3509` / `0.5000` / `0.3953`
- F1 idle/left/right：`0.4721` / `0.0150` / `0.3849`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     93      1     35
  true1     87      1     43
  true2     85      0     51
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3643`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3171`
- Val loss（最优时）：`1.0977`

**Test（overall）**
- Accuracy：`0.3794`
- Balanced Acc：`0.3779`
- F1-macro：`0.3731`
- Precision-macro：`0.3771`
- Recall-macro：`0.3779`
- Recall idle/left/right：`0.2868` / `0.5111` / `0.3358`
- Precision idle/left/right：`0.3814` / `0.4012` / `0.3488`
- F1 idle/left/right：`0.3274` / `0.4495` / `0.3422`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     37     46     46
  true1     28     69     38
  true2     32     57     45
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`117`
- 验证最优轮次（best_epoch）：`99`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4759`
- Val F1-macro（最优 checkpoint 时，附报）：`0.4271`
- Val loss（最优时）：`1.0633`

**Test（overall）**
- Accuracy：`0.4026`
- Balanced Acc：`0.3994`
- F1-macro：`0.3460`
- Precision-macro：`0.4302`
- Recall-macro：`0.3994`
- Recall idle/left/right：`0.0625` / `0.4538` / `0.6818`
- Precision idle/left/right：`0.5000` / `0.3758` / `0.4147`
- F1 idle/left/right：`0.1111` / `0.4111` / `0.5158`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      8     60     60
  true1      4     59     67
  true2      4     38     90
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4456`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3437`
- Val loss（最优时）：`1.0946`

**Test（overall）**
- Accuracy：`0.3560`
- Balanced Acc：`0.3640`
- F1-macro：`0.2673`
- Precision-macro：`0.2479`
- Recall-macro：`0.3640`
- Recall idle/left/right：`0.8319` / `0.2602` / `0.0000`
- Precision idle/left/right：`0.3438` / `0.4000` / `0.0000`
- F1 idle/left/right：`0.4865` / `0.3153` / `0.0000`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     99     20      0
  true1     91     32      0
  true2     98     28      0
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3410`
- Val F1-macro（最优 checkpoint 时，附报）：`0.1809`
- Val loss（最优时）：`1.1077`

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

- 结束：`2026-08-04T10:39:31`
