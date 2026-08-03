# 被试独立五折实验记录（20260802_134219 / eegnet_stieger_4s）

- 开始：`2026-08-02T13:42:19`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\stieger_4s`（prefix=`stieger`，tag=`stieger_4s`）
- 切窗说明：反馈段最后 4s（Task/Rest 同取法）；1000@250Hz
- model：`eegnet_stieger_4s`（EEGNet 普通 CE；**仅 Task**；无加权/无采样平衡/无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`1000`
- 样本：N=`6592` Rest=`1722` Task=`4870`
- 划分：被试独立五折（非 LOSO、非混合库）
- 早停：Val F1（与 baseline_eegnet 一致）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_stieger_4s\stieger_4s\run_20260802_134219`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val F1（最优）：`0.8651`
- Val loss（最优时）：`0.5607`

**Test（overall）**
- Accuracy：`0.7282`
- Recall：`0.9748`
- Specificity：`0.0519`
- Precision：`0.7382`
- F1：`0.8402`
- Balanced Acc：`0.5133`
- 混淆矩阵：TP=`1083` TN=`21` FP=`384` FN=`28`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`33`
- 验证最优轮次（best_epoch）：`15`
- Val F1（最优）：`0.8105`
- Val loss（最优时）：`0.6248`

**Test（overall）**
- Accuracy：`0.8039`
- Recall：`0.9906`
- Specificity：`0.1387`
- Precision：`0.8038`
- F1：`0.8875`
- Balanced Acc：`0.5646`
- 混淆矩阵：TP=`840` TN=`33` FP=`205` FN=`8`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`27`
- 验证最优轮次（best_epoch）：`9`
- Val F1（最优）：`0.8105`
- Val loss（最优时）：`0.6399`

**Test（overall）**
- Accuracy：`0.7586`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.7586`
- F1：`0.8627`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`1015` TN=`0` FP=`323` FN=`0`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val F1（最优）：`0.8664`
- Val loss（最优时）：`0.5509`

**Test（overall）**
- Accuracy：`0.6810`
- Recall：`0.9942`
- Specificity：`0.0124`
- Precision：`0.6824`
- F1：`0.8093`
- Balanced Acc：`0.5033`
- 混淆矩阵：TP=`1025` TN=`6` FP=`477` FN=`6`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val F1（最优）：`0.8102`
- Val loss（最优时）：`0.6757`

**Test（overall）**
- Accuracy：`0.7601`
- Recall：`1.0000`
- Specificity：`0.0000`
- Precision：`0.7601`
- F1：`0.8637`
- Balanced Acc：`0.5000`
- 混淆矩阵：TP=`865` TN=`0` FP=`273` FN=`0`

## 最终结论（Test 五折均值）

- Val F1：`0.8326 ± 0.0271`
- Test Acc：`0.7464 ± 0.0406`
- Test Spec：`0.0406 ± 0.0526`
- Test Rec：`0.9919 ± 0.0093`
- Test Precision：`0.7486 ± 0.0394`
- Test F1：`0.8527 ± 0.0263`
- Test BalAcc：`0.5162 ± 0.0247`

- 结束：`2026-08-02T13:50:50`
