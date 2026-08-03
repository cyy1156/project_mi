# 被试独立五折实验记录（20260802_161757 / eegnet_bci2a_2s_wce2p2_balacc）

- 开始：`2026-08-02T16:17:57`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，tag=`bci2a_2s`）
- 切窗说明：任务 Cue+2~4s；静息下一 Cue 前 2s；500@250Hz
- model：`eegnet_bci2a_2s_wce2p2_balacc`（EEGNet 加权 CE w0=2.2/w1=1.0；**仅 Task**；无 batch balance / 无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`500`
- 样本：N=`1719` Rest=`556` Task=`1163`
- 划分：被试独立五折（非 LOSO、非混合库）
- train sampler：无（普通 shuffle）
- 损失：固定加权 CE，静息 w0=`2.2`，任务 w1=`1.0`
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_bci2a_2s_wce2p2_balacc\bci2a_2s\run_20260802_161757`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`62`
- 验证最优轮次（best_epoch）：`44`
- Val 选模分数（Balanced Acc）：`0.7247`
- Val F1（最优 checkpoint 时，附报）：`0.7097`
- Val loss（最优时）：`0.5838`

**Test（overall）**
- Accuracy：`0.6061`
- Recall：`0.5880`
- Specificity：`0.6434`
- Precision：`0.7734`
- F1：`0.6681`
- Balanced Acc：`0.6157`
- 混淆矩阵：TP=`157` TN=`83` FP=`46` FN=`110`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`70`
- 验证最优轮次（best_epoch）：`52`
- Val 选模分数（Balanced Acc）：`0.7040`
- Val F1（最优 checkpoint 时，附报）：`0.7123`
- Val loss（最优时）：`0.6561`

**Test（overall）**
- Accuracy：`0.6759`
- Recall：`0.6134`
- Specificity：`0.8062`
- Precision：`0.8684`
- F1：`0.7190`
- Balanced Acc：`0.7098`
- 混淆矩阵：TP=`165` TN=`104` FP=`25` FN=`104`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`65`
- 验证最优轮次（best_epoch）：`47`
- Val 选模分数（Balanced Acc）：`0.6836`
- Val F1（最优 checkpoint 时，附报）：`0.7742`
- Val loss（最优时）：`0.6384`

**Test（overall）**
- Accuracy：`0.6410`
- Recall：`0.7176`
- Specificity：`0.4844`
- Precision：`0.7402`
- F1：`0.7287`
- Balanced Acc：`0.6010`
- 混淆矩阵：TP=`188` TN=`62` FP=`66` FN=`74`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc）：`0.7001`
- Val F1（最优 checkpoint 时，附报）：`0.7064`
- Val loss（最优时）：`0.6575`

**Test（overall）**
- Accuracy：`0.4592`
- Recall：`0.3855`
- Specificity：`0.6134`
- Precision：`0.6761`
- F1：`0.4910`
- Balanced Acc：`0.4995`
- 混淆矩阵：TP=`96` TN=`73` FP=`46` FN=`153`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.6893`
- Val F1（最优 checkpoint 时，附报）：`0.7458`
- Val loss（最优时）：`0.6605`

**Test（overall）**
- Accuracy：`0.6647`
- Recall：`0.6293`
- Specificity：`0.7451`
- Precision：`0.8488`
- F1：`0.7228`
- Balanced Acc：`0.6872`
- 混淆矩阵：TP=`73` TN=`38` FP=`13` FN=`43`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.7003 ± 0.0142`
- Val F1（附报）：`0.7297 ± 0.0264`
- Test Acc：`0.6094 ± 0.0788`
- Test Spec：`0.6585 ± 0.1113`
- Test Rec：`0.5868 ± 0.1097`
- Test Precision：`0.7814 ± 0.0707`
- Test F1：`0.6659 ± 0.0901`
- Test BalAcc：`0.6226 ± 0.0741`

- 结束：`2026-08-02T16:19:56`
