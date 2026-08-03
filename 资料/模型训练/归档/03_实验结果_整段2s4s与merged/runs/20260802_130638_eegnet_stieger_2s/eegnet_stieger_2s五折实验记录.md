# 被试独立五折实验记录（20260802_130638 / eegnet_stieger_2s）

- 开始：`2026-08-02T13:06:38`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\stieger_2s`（prefix=`stieger`，tag=`stieger_2s`）
- 切窗说明：反馈段最后 2s（Task/Rest 同取法）；500@250Hz
- model：`eegnet_stieger_2s`（EEGNet 普通 CE；**仅 Task**；无加权/无采样平衡/无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`500`
- 样本：N=`34337` Rest=`9775` Task=`24562`
- 划分：被试独立五折（非 LOSO、非混合库）
- 早停：Val F1（与 baseline_eegnet 一致）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_stieger_2s\stieger_2s\run_20260802_130638`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val F1（最优）：`0.8483`
- Val loss（最优时）：`0.6010`

**Test（overall）**
- Accuracy：`0.7133`
- Recall：`0.9975`
- Specificity：`0.0037`
- Precision：`0.7142`
- F1：`0.8324`
- Balanced Acc：`0.5006`
- 混淆矩阵：TP=`4769` TN=`7` FP=`1908` FN=`12`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val F1（最优）：`0.8099`
- Val loss（最优时）：`0.6255`

**Test（overall）**
- Accuracy：`0.7256`
- Recall：`0.9856`
- Specificity：`0.0531`
- Precision：`0.7292`
- F1：`0.8382`
- Balanced Acc：`0.5194`
- 混淆矩阵：TP=`4031` TN=`84` FP=`1497` FN=`59`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val F1（最优）：`0.8291`
- Val loss（最优时）：`0.6222`

**Test（overall）**
- Accuracy：`0.7607`
- Recall：`0.9969`
- Specificity：`0.0483`
- Precision：`0.7595`
- F1：`0.8622`
- Balanced Acc：`0.5226`
- 混淆矩阵：TP=`4479` TN=`72` FP=`1418` FN=`14`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val F1（最优）：`0.8317`
- Val loss（最优时）：`0.6095`

**Test（overall）**
- Accuracy：`0.6926`
- Recall：`0.9996`
- Specificity：`0.0008`
- Precision：`0.6927`
- F1：`0.8183`
- Balanced Acc：`0.5002`
- 混淆矩阵：TP=`5418` TN=`2` FP=`2404` FN=`2`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val F1（最优）：`0.8438`
- Val loss（最优时）：`0.5941`

**Test（overall）**
- Accuracy：`0.7075`
- Recall：`0.9991`
- Specificity：`0.0004`
- Precision：`0.7079`
- F1：`0.8287`
- Balanced Acc：`0.4998`
- 混淆矩阵：TP=`5773` TN=`1` FP=`2382` FN=`5`

## 最终结论（Test 五折均值）

- Val F1：`0.8326 ± 0.0134`
- Test Acc：`0.7199 ± 0.0230`
- Test Spec：`0.0213 ± 0.0241`
- Test Rec：`0.9957 ± 0.0052`
- Test Precision：`0.7207 ± 0.0227`
- Test F1：`0.8360 ± 0.0146`
- Test BalAcc：`0.5085 ± 0.0102`

- 结束：`2026-08-02T13:23:08`
