# 被试独立五折实验记录（20260803_180949 / dbn_1s_balbatch_balacc）

- 开始：`2026-08-03T18:09:49`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_1s`（prefix=`bci2a`，**单库不合并**）
- protocol：`1s-offline-native-arch` | **no_rap=True**
- model：`dbn`（原结构）
- 结构：DBN + 1s μ/β log bandpower (N,8,2)
- 早停：Val Balanced Accuracy | 训练采样：batch balance 1:1
- shared hp：`{'data_tag': 'bci2a_1s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5, 'protocol': '1s-offline-native-arch', 'early_stop': 'balanced_accuracy', 'train_sampler': 'balanced_invfreq', 'n_times_expected': 250, 'no_rap': True}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline_1s\dbn_1s_balbatch_balacc\bci2a_1s\run_20260803_180949`

---
## 最终结论（Task only）

- Val BalAcc：`0.5605 ± 0.0654`
- Test BalAcc：`0.5191 ± 0.0114`
- Test Spec：`0.6039 ± 0.3222`
- Test Rec：`0.4344 ± 0.3419`
- Test F1：`0.4517 ± 0.2822`
- Test Acc：`0.4842 ± 0.1335`

### Task 各折明细

说明：早停与选模均为 **Val Balanced Accuracy**；训练集使用 **batch balance**（WeightedRandomSampler）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.6910`
- Val F1（最优 checkpoint 时，附报）：`0.7453`
- Val loss（最优时）：`0.6808`

**Test（overall）**
- Accuracy：`0.6709`
- Recall：`0.9202`
- Specificity：`0.1462`
- Precision：`0.6940`
- F1：`0.7912`
- Balanced Acc：`0.5332`
- 混淆矩阵：TP=`18672` TN=`1410` FP=`8233` FN=`1620`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`35`
- 验证最优轮次（best_epoch）：`17`
- Val 选模分数（Balanced Acc）：`0.5286`
- Val F1（最优 checkpoint 时，附报）：`0.4835`
- Val loss（最优时）：`0.7111`

**Test（overall）**
- Accuracy：`0.3969`
- Recall：`0.1875`
- Specificity：`0.8428`
- Precision：`0.7174`
- F1：`0.2973`
- Balanced Acc：`0.5151`
- 混淆矩阵：TP=`3833` TN=`8094` FP=`1510` FN=`16611`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val 选模分数（Balanced Acc）：`0.5303`
- Val F1（最优 checkpoint 时，附报）：`0.7370`
- Val loss（最优时）：`0.6659`

**Test（overall）**
- Accuracy：`0.5992`
- Recall：`0.7380`
- Specificity：`0.3094`
- Precision：`0.6905`
- F1：`0.7135`
- Balanced Acc：`0.5237`
- 混淆矩阵：TP=`14695` TN=`2951` FP=`6587` FN=`5217`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val 选模分数（Balanced Acc）：`0.5316`
- Val F1（最优 checkpoint 时，附报）：`0.7345`
- Val loss（最优时）：`0.6846`

**Test（overall）**
- Accuracy：`0.4496`
- Recall：`0.3185`
- Specificity：`0.7301`
- Precision：`0.7163`
- F1：`0.4409`
- Balanced Acc：`0.5243`
- 混淆矩阵：TP=`6027` TN=`6456` FP=`2387` FN=`12897`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5208`
- Val F1（最优 checkpoint 时，附报）：`0.7923`
- Val loss（最优时）：`0.6681`

**Test（overall）**
- Accuracy：`0.3046`
- Recall：`0.0078`
- Specificity：`0.9911`
- Precision：`0.6699`
- F1：`0.0155`
- Balanced Acc：`0.4995`
- 混淆矩阵：TP=`69` TN=`3778` FP=`34` FN=`8747`

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

- 结束：`2026-08-03T18:19:48`
