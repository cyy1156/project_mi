# 被试独立五折实验记录（20260802_162406 / eegnet_stieger_2s_wce2p2_balacc）

- 开始：`2026-08-02T16:24:06`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\stieger_2s`（prefix=`stieger`，tag=`stieger_2s`）
- 切窗说明：反馈段最后 2s（Task/Rest 同取法）；500@250Hz
- model：`eegnet_stieger_2s_wce2p2_balacc`（EEGNet 加权 CE w0=2.2/w1=1.0；**仅 Task**；无 batch balance / 无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`500`
- 样本：N=`34337` Rest=`9775` Task=`24562`
- 划分：被试独立五折（非 LOSO、非混合库）
- train sampler：无（普通 shuffle）
- 损失：固定加权 CE，静息 w0=`2.2`，任务 w1=`1.0`
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_stieger_2s_wce2p2_balacc\stieger_2s\run_20260802_162406`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.6132`
- Val F1（最优 checkpoint 时，附报）：`0.7148`
- Val loss（最优时）：`0.6774`

**Test（overall）**
- Accuracy：`0.6304`
- Recall：`0.7279`
- Specificity：`0.3869`
- Precision：`0.7477`
- F1：`0.7377`
- Balanced Acc：`0.5574`
- 混淆矩阵：TP=`3480` TN=`741` FP=`1174` FN=`1301`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val 选模分数（Balanced Acc）：`0.5705`
- Val F1（最优 checkpoint 时，附报）：`0.7552`
- Val loss（最优时）：`0.6767`

**Test（overall）**
- Accuracy：`0.4555`
- Recall：`0.3457`
- Specificity：`0.7394`
- Precision：`0.7744`
- F1：`0.4780`
- Balanced Acc：`0.5426`
- 混淆矩阵：TP=`1414` TN=`1169` FP=`412` FN=`2676`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5561`
- Val F1（最优 checkpoint 时，附报）：`0.6643`
- Val loss（最优时）：`0.6857`

**Test（overall）**
- Accuracy：`0.7115`
- Recall：`0.7474`
- Specificity：`0.6034`
- Precision：`0.8503`
- F1：`0.7955`
- Balanced Acc：`0.6754`
- 混淆矩阵：TP=`3358` TN=`899` FP=`591` FN=`1135`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`37`
- 验证最优轮次（best_epoch）：`19`
- Val 选模分数（Balanced Acc）：`0.6156`
- Val F1（最优 checkpoint 时，附报）：`0.7002`
- Val loss（最优时）：`0.6595`

**Test（overall）**
- Accuracy：`0.6393`
- Recall：`0.8004`
- Specificity：`0.2764`
- Precision：`0.7136`
- F1：`0.7545`
- Balanced Acc：`0.5384`
- 混淆矩阵：TP=`4338` TN=`665` FP=`1741` FN=`1082`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`33`
- Val 选模分数（Balanced Acc）：`0.6102`
- Val F1（最优 checkpoint 时，附报）：`0.6895`
- Val loss（最优时）：`0.6722`

**Test（overall）**
- Accuracy：`0.6282`
- Recall：`0.7193`
- Specificity：`0.4075`
- Precision：`0.7464`
- F1：`0.7326`
- Balanced Acc：`0.5634`
- 混淆矩阵：TP=`4156` TN=`971` FP=`1412` FN=`1622`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.5931 ± 0.0248`
- Val F1（附报）：`0.7048 ± 0.0301`
- Test Acc：`0.6130 ± 0.0846`
- Test Spec：`0.4827 ± 0.1660`
- Test Rec：`0.6681 ± 0.1636`
- Test Precision：`0.7665 ± 0.0461`
- Test F1：`0.6997 ± 0.1130`
- Test BalAcc：`0.5754 ± 0.0508`

- 结束：`2026-08-02T16:51:41`
