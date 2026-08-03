# 被试独立五折实验记录（20260802_130350 / eegnet_bci2a_4s）

- 开始：`2026-08-02T13:03:50`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_4s`（prefix=`bci2a`，tag=`bci2a_4s`）
- 切窗说明：任务 Cue+0~4s；静息下一 Cue 前 4s；1000@250Hz
- model：`eegnet_bci2a_4s`（EEGNet 普通 CE；**仅 Task**；无加权/无采样平衡/无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`1000`
- 样本：N=`1719` Rest=`556` Task=`1163`
- 划分：被试独立五折（非 LOSO、非混合库）
- 早停：Val F1（与 baseline_eegnet 一致）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_bci2a_4s\bci2a_4s\run_20260802_130350`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`26`
- 验证最优轮次（best_epoch）：`8`
- Val F1（最优）：`0.8078`
- Val loss（最优时）：`0.6400`

**Test（overall）**
- Accuracy：`0.6995`
- Recall：`0.9775`
- Specificity：`0.1240`
- Precision：`0.6979`
- F1：`0.8144`
- Balanced Acc：`0.5508`
- 混淆矩阵：TP=`261` TN=`16` FP=`113` FN=`6`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`20`
- 验证最优轮次（best_epoch）：`2`
- Val F1（最优）：`0.8113`
- Val loss（最优时）：`0.6860`

**Test（overall）**
- Accuracy：`0.6608`
- Recall：`0.9740`
- Specificity：`0.0078`
- Precision：`0.6718`
- F1：`0.7951`
- Balanced Acc：`0.4909`
- 混淆矩阵：TP=`262` TN=`1` FP=`128` FN=`7`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`75`
- 验证最优轮次（best_epoch）：`57`
- Val F1（最优）：`0.8493`
- Val loss（最优时）：`0.4946`

**Test（overall）**
- Accuracy：`0.7718`
- Recall：`0.9733`
- Specificity：`0.3594`
- Precision：`0.7567`
- F1：`0.8514`
- Balanced Acc：`0.6663`
- 混淆矩阵：TP=`255` TN=`46` FP=`82` FN=`7`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`31`
- 验证最优轮次（best_epoch）：`13`
- Val F1（最优）：`0.8285`
- Val loss（最优时）：`0.6047`

**Test（overall）**
- Accuracy：`0.6576`
- Recall：`0.8072`
- Specificity：`0.3445`
- Precision：`0.7204`
- F1：`0.7614`
- Balanced Acc：`0.5759`
- 混淆矩阵：TP=`201` TN=`41` FP=`78` FN=`48`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`52`
- 验证最优轮次（best_epoch）：`34`
- Val F1（最优）：`0.8615`
- Val loss（最优时）：`0.5021`

**Test（overall）**
- Accuracy：`0.8503`
- Recall：`0.9397`
- Specificity：`0.6471`
- Precision：`0.8583`
- F1：`0.8971`
- Balanced Acc：`0.7934`
- 混淆矩阵：TP=`109` TN=`33` FP=`18` FN=`7`

## 最终结论（Test 五折均值）

- Val F1：`0.8317 ± 0.0210`
- Test Acc：`0.7280 ± 0.0737`
- Test Spec：`0.2966 ± 0.2201`
- Test Rec：`0.9343 ± 0.0650`
- Test Precision：`0.7410 ± 0.0649`
- Test F1：`0.8239 ± 0.0468`
- Test BalAcc：`0.6154 ± 0.1054`

- 结束：`2026-08-02T13:06:37`
