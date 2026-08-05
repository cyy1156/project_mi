# 被试独立五折实验记录（20260804_103311 / gcbnet_fixed2s_balbatch_balacc）

- 开始：`2026-08-04T10:33:11`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，**仅 BCI2a；不合并**）
- protocol：`fixed-2s-cue2to4-bci2a` | **no_rap=True**
- 读数口径：`Tw=2s fixed cue+2_4s rest_cue_before_2s bci2a_only balacc+balbatch no_rap`
- 切窗：Task=Cue+2~4s；Rest=Cue前2s；**固定窗无 hop**
- model：`gcbnet`（原结构）
- 结构：GCBNet(k=2, layers=[128]) + 2s μ/β log bandpower (N,8,2)
- 流程：Task（静息/任务）→ Three（空闲/左/右，**独立重训，不迁移权重**）
- 早停：Val Balanced Accuracy | 训练采样：batch balance（Task 1:1 / Three 三类）
- shared hp：`{'data_tag': 'bci2a_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': 'fixed-2s-cue2to4-bci2a', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 500, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_fixed_2s\gcbnet_fixed2s_balbatch_balacc\bci2a_2s\run_20260804_103311`

---
## 最终结论

### Task（静息/任务）
- Val BalAcc：`0.5923 ± 0.0949`
- Test BalAcc：`0.5543 ± 0.0454`
- Test Spec：`0.5115 ± 0.3217`
- Test Rec：`0.5971 ± 0.3147`
- Test F1：`0.5906 ± 0.2515`
- Test Acc：`0.5650 ± 0.1219`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.7807`
- Val F1（最优 checkpoint 时，附报）：`0.8098`
- Val loss（最优时）：`0.6761`

**Test（overall）**
- Accuracy：`0.6187`
- Recall：`0.8614`
- Specificity：`0.1163`
- Precision：`0.6686`
- F1：`0.7529`
- Balanced Acc：`0.4889`
- 混淆矩阵：TP=`230` TN=`15` FP=`114` FN=`37`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`69`
- 验证最优轮次（best_epoch）：`51`
- Val 选模分数（Balanced Acc）：`0.5571`
- Val F1（最优 checkpoint 时，附报）：`0.7805`
- Val loss（最优时）：`0.6518`

**Test（overall）**
- Accuracy：`0.5226`
- Recall：`0.4535`
- Specificity：`0.6667`
- Precision：`0.7394`
- F1：`0.5622`
- Balanced Acc：`0.5601`
- 混淆矩阵：TP=`122` TN=`86` FP=`43` FN=`147`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`107`
- 验证最优轮次（best_epoch）：`89`
- Val 选模分数（Balanced Acc）：`0.5548`
- Val F1（最优 checkpoint 时，附报）：`0.8079`
- Val loss（最优时）：`0.6186`

**Test（overall）**
- Accuracy：`0.6897`
- Recall：`0.9275`
- Specificity：`0.2031`
- Precision：`0.7043`
- F1：`0.8007`
- Balanced Acc：`0.5653`
- 混淆矩阵：TP=`243` TN=`26` FP=`102` FN=`19`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc）：`0.5255`
- Val F1（最优 checkpoint 时，附报）：`0.8065`
- Val loss（最优时）：`0.6698`

**Test（overall）**
- Accuracy：`0.6467`
- Recall：`0.6827`
- Specificity：`0.5714`
- Precision：`0.7692`
- F1：`0.7234`
- Balanced Acc：`0.6271`
- 混淆矩阵：TP=`170` TN=`68` FP=`51` FN=`79`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5433`
- Val F1（最优 checkpoint 时，附报）：`0.7469`
- Val loss（最优时）：`0.6822`

**Test（overall）**
- Accuracy：`0.3473`
- Recall：`0.0603`
- Specificity：`1.0000`
- Precision：`1.0000`
- F1：`0.1138`
- Balanced Acc：`0.5302`
- 混淆矩阵：TP=`7` TN=`51` FP=`0` FN=`109`

### Three（空闲/左/右；独立重训，不使用 Task 权重）
- Val BalAcc：`0.4371 ± 0.0711`
- Val F1-macro：`0.3872 ± 0.0988`
- Test BalAcc：`0.3866 ± 0.0459`
- Test F1-macro：`0.3265 ± 0.0970`
- Test Acc：`0.3825 ± 0.0511`
- Test Precision-macro：`0.3989 ± 0.1002`
- Test Recall-macro：`0.3866 ± 0.0459`
- Test Recall idle/left/right：`0.4060±0.3733` / `0.2930±0.1676` / `0.4609±0.2305`

### Three 各折明细

说明：与 Task **独立新建模型**（不迁移权重）；早停/选模为 **Val Balanced Acc（=Recall-macro）**；训练 **batch balance**（三类 inverse-freq）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`135`
- 验证最优轮次（best_epoch）：`117`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.5722`
- Val F1-macro（最优 checkpoint 时，附报）：`0.5706`
- Val loss（最优时）：`1.0220`

**Test（overall）**
- Accuracy：`0.3788`
- Balanced Acc：`0.3750`
- F1-macro：`0.3323`
- Precision-macro：`0.4074`
- Recall-macro：`0.3750`
- Recall idle/left/right：`0.0698` / `0.5038` / `0.5515`
- Precision idle/left/right：`0.4737` / `0.3771` / `0.3713`
- F1 idle/left/right：`0.1216` / `0.4314` / `0.4438`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      9     50     70
  true1      8     66     57
  true2      2     59     75
```

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`64`
- 验证最优轮次（best_epoch）：`46`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3991`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3760`
- Val loss（最优时）：`1.0993`

**Test（overall）**
- Accuracy：`0.4698`
- Balanced Acc：`0.4727`
- F1-macro：`0.4603`
- Precision-macro：`0.5093`
- Recall-macro：`0.4727`
- Recall idle/left/right：`0.6589` / `0.2593` / `0.5000`
- Precision idle/left/right：`0.3829` / `0.5469` / `0.5982`
- F1 idle/left/right：`0.4843` / `0.3518` / `0.5447`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     85     13     31
  true1     86     35     14
  true2     51     16     67
```

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4373`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3811`
- Val loss（最优时）：`1.0973`

**Test（overall）**
- Accuracy：`0.3641`
- Balanced Acc：`0.3607`
- F1-macro：`0.2919`
- Precision-macro：`0.4627`
- Recall-macro：`0.3607`
- Recall idle/left/right：`0.0156` / `0.3846` / `0.6818`
- Precision idle/left/right：`0.6667` / `0.3571` / `0.3644`
- F1 idle/left/right：`0.0305` / `0.3704` / `0.4749`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0      2     49     77
  true1      0     50     80
  true2      1     41     90
```

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.4092`
- Val F1-macro（最优 checkpoint 时，附报）：`0.3290`
- Val loss（最优时）：`1.0939`

**Test（overall）**
- Accuracy：`0.3886`
- Balanced Acc：`0.3861`
- F1-macro：`0.3798`
- Precision-macro：`0.4003`
- Recall-macro：`0.3861`
- Recall idle/left/right：`0.2857` / `0.3171` / `0.5556`
- Precision idle/left/right：`0.4722` / `0.3482` / `0.3804`
- F1 idle/left/right：`0.3560` / `0.3319` / `0.4516`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     34     38     47
  true1     17     39     67
  true2     21     35     70
```

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`25`
- 验证最优轮次（best_epoch）：`7`
- Val 选模分数（Balanced Acc = Recall-macro）：`0.3675`
- Val F1-macro（最优 checkpoint 时，附报）：`0.2793`
- Val loss（最优时）：`1.1030`

**Test（overall）**
- Accuracy：`0.3114`
- Balanced Acc：`0.3386`
- F1-macro：`0.1682`
- Precision-macro：`0.2148`
- Recall-macro：`0.3386`
- Recall idle/left/right：`1.0000` / `0.0000` / `0.0159`
- Precision idle/left/right：`0.3110` / `0.0000` / `0.3333`
- F1 idle/left/right：`0.4744` / `0.0000` / `0.0303`
- 混淆矩阵（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0     51      0      0
  true1     51      0      2
  true2     62      0      1
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

- 结束：`2026-08-04T10:34:31`
