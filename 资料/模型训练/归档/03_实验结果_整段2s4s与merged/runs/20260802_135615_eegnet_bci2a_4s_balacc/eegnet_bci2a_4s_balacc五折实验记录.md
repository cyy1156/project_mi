# 被试独立五折实验记录（20260802_135615 / eegnet_bci2a_4s_balacc）

- 开始：`2026-08-02T13:56:15`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_4s`（prefix=`bci2a`，tag=`bci2a_4s`）
- 切窗说明：任务 Cue+0~4s；静息下一 Cue 前 4s；1000@250Hz
- model：`eegnet_bci2a_4s_balacc`（EEGNet 普通 CE；**仅 Task**；无加权/无采样平衡/无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`1000`
- 样本：N=`1719` Rest=`556` Task=`1163`
- 划分：被试独立五折（非 LOSO、非混合库）
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_bci2a_4s_balacc\bci2a_4s\run_20260802_135615`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`142`
- 验证最优轮次（best_epoch）：`124`
- Val 选模分数（Balanced Acc）：`0.8509`
- Val F1（最优 checkpoint 时，附报）：`0.8585`
- Val loss（最优时）：`0.4471`

**Test（overall）**
- Accuracy：`0.7475`
- Recall：`0.9251`
- Specificity：`0.3798`
- Precision：`0.7554`
- F1：`0.8316`
- Balanced Acc：`0.6525`
- 混淆矩阵：TP=`247` TN=`49` FP=`80` FN=`20`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5229`
- Val F1（最优 checkpoint 时，附报）：`0.7867`
- Val loss（最优时）：`0.6902`

**Test（overall）**
- Accuracy：`0.6156`
- Recall：`0.8736`
- Specificity：`0.0775`
- Precision：`0.6638`
- F1：`0.7544`
- Balanced Acc：`0.4756`
- 混淆矩阵：TP=`235` TN=`10` FP=`119` FN=`34`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`79`
- 验证最优轮次（best_epoch）：`61`
- Val 选模分数（Balanced Acc）：`0.6614`
- Val F1（最优 checkpoint 时，附报）：`0.8443`
- Val loss（最优时）：`0.4913`

**Test（overall）**
- Accuracy：`0.7744`
- Recall：`0.9695`
- Specificity：`0.3750`
- Precision：`0.7605`
- F1：`0.8523`
- Balanced Acc：`0.6722`
- 混淆矩阵：TP=`254` TN=`48` FP=`80` FN=`8`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`64`
- 验证最优轮次（best_epoch）：`46`
- Val 选模分数（Balanced Acc）：`0.6986`
- Val F1（最优 checkpoint 时，附报）：`0.8000`
- Val loss（最优时）：`0.5629`

**Test（overall）**
- Accuracy：`0.5571`
- Recall：`0.5823`
- Specificity：`0.5042`
- Precision：`0.7108`
- F1：`0.6402`
- Balanced Acc：`0.5433`
- 混淆矩阵：TP=`145` TN=`60` FP=`59` FN=`104`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`52`
- 验证最优轮次（best_epoch）：`34`
- Val 选模分数（Balanced Acc）：`0.7036`
- Val F1（最优 checkpoint 时，附报）：`0.8615`
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

- Val BalAcc（选模）：`0.6875 ± 0.1047`
- Val F1（附报）：`0.8302 ± 0.0310`
- Test Acc：`0.7090 ± 0.1073`
- Test Spec：`0.3967 ± 0.1881`
- Test Rec：`0.8580 ± 0.1413`
- Test Precision：`0.7497 ± 0.0645`
- Test F1：`0.7951 ± 0.0902`
- Test BalAcc：`0.6274 ± 0.1098`

- 结束：`2026-08-02T14:01:25`
