# 被试独立五折实验记录（20260803_185925 / dgcnn_1s_balbatch_balacc）

- 开始：`2026-08-03T18:59:25`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`dgcnn`（原结构）
- 结构：DGCNN(k=2, layers=[128]) + 1s bandpower
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\dgcnn_1s_balbatch_balacc\bci2a_1s\run_20260803_185925`

---
## 最终结论（Task only）

- Val BalAcc：`0.5487 ± 0.0496`
- Test BalAcc：`0.5382 ± 0.0103`
- Test Spec：`0.3679 ± 0.1604`
- Test Rec：`0.7084 ± 0.1485`
- Test F1：`0.6999 ± 0.0713`
- Test Acc：`0.6005 ± 0.0502`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.6460`
- Val F1（最优 checkpoint 时，附报）：`0.8089`
- Val loss（最优时）：`0.6562`

**Test（overall）**
- Accuracy：`0.6226`
- Recall：`0.7535`
- Specificity：`0.3471`
- Precision：`0.7083`
- F1：`0.7302`
- Balanced Acc：`0.5503`
- 混淆矩阵：TP=`15290` TN=`3347` FP=`6296` FN=`5002`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5304`
- Val F1（最优 checkpoint 时，附报）：`0.7650`
- Val loss（最优时）：`0.6658`

**Test（overall）**
- Accuracy：`0.5273`
- Recall：`0.5119`
- Specificity：`0.5600`
- Precision：`0.7123`
- F1：`0.5957`
- Balanced Acc：`0.5359`
- 混淆矩阵：TP=`10465` TN=`5378` FP=`4226` FN=`9979`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5374`
- Val F1（最优 checkpoint 时，附报）：`0.8003`
- Val loss（最优时）：`0.6431`

**Test（overall）**
- Accuracy：`0.6645`
- Recall：`0.9240`
- Specificity：`0.1228`
- Precision：`0.6874`
- F1：`0.7883`
- Balanced Acc：`0.5234`
- 混淆矩阵：TP=`18399` TN=`1171` FP=`8367` FN=`1513`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5089`
- Val F1（最优 checkpoint 时，附报）：`0.6943`
- Val loss（最优时）：`0.6881`

**Test（overall）**
- Accuracy：`0.5581`
- Recall：`0.5739`
- Specificity：`0.5243`
- Precision：`0.7208`
- F1：`0.6390`
- Balanced Acc：`0.5491`
- 混淆矩阵：TP=`10860` TN=`4636` FP=`4207` FN=`8064`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`39`
- 验证最优轮次（best_epoch）：`21`
- Val 选模分数（Balanced Acc）：`0.5208`
- Val F1（最优 checkpoint 时，附报）：`0.7464`
- Val loss（最优时）：`0.6717`

**Test（overall）**
- Accuracy：`0.6299`
- Recall：`0.7788`
- Specificity：`0.2854`
- Precision：`0.7160`
- F1：`0.7461`
- Balanced Acc：`0.5321`
- 混淆矩阵：TP=`6866` TN=`1088` FP=`2724` FN=`1950`

### 共用超参
```json
{
  "data_tag": "bci2a_1s",
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
  "protocol": "1s-offline-native-arch",
  "early_stop": "balanced_accuracy",
  "train_sampler": "balanced_invfreq",
  "n_times_expected": 250,
  "no_rap": true
}
```

- 结束：`2026-08-03T19:17:46`
