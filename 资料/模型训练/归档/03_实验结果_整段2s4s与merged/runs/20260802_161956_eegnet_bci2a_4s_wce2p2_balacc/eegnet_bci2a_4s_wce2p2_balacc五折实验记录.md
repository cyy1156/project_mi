# 被试独立五折实验记录（20260802_161956 / eegnet_bci2a_4s_wce2p2_balacc）

- 开始：`2026-08-02T16:19:56`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_4s`（prefix=`bci2a`，tag=`bci2a_4s`）
- 切窗说明：任务 Cue+0~4s；静息下一 Cue 前 4s；1000@250Hz
- model：`eegnet_bci2a_4s_wce2p2_balacc`（EEGNet 加权 CE w0=2.2/w1=1.0；**仅 Task**；无 batch balance / 无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`1000`
- 样本：N=`1719` Rest=`556` Task=`1163`
- 划分：被试独立五折（非 LOSO、非混合库）
- train sampler：无（普通 shuffle）
- 损失：固定加权 CE，静息 w0=`2.2`，任务 w1=`1.0`
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_bci2a_4s_wce2p2_balacc\bci2a_4s\run_20260802_161956`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`142`
- 验证最优轮次（best_epoch）：`124`
- Val 选模分数（Balanced Acc）：`0.8251`
- Val F1（最优 checkpoint 时，附报）：`0.8163`
- Val loss（最优时）：`0.4415`

**Test（overall）**
- Accuracy：`0.7348`
- Recall：`0.8352`
- Specificity：`0.5271`
- Precision：`0.7852`
- F1：`0.8094`
- Balanced Acc：`0.6812`
- 混淆矩阵：TP=`223` TN=`68` FP=`61` FN=`44`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`62`
- 验证最优轮次（best_epoch）：`44`
- Val 选模分数（Balanced Acc）：`0.7193`
- Val F1（最优 checkpoint 时，附报）：`0.8048`
- Val loss（最优时）：`0.6006`

**Test（overall）**
- Accuracy：`0.7739`
- Recall：`0.7435`
- Specificity：`0.8372`
- Precision：`0.9050`
- F1：`0.8163`
- Balanced Acc：`0.7904`
- 混淆矩阵：TP=`200` TN=`108` FP=`21` FN=`69`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`44`
- 验证最优轮次（best_epoch）：`26`
- Val 选模分数（Balanced Acc）：`0.7202`
- Val F1（最优 checkpoint 时，附报）：`0.7918`
- Val loss（最优时）：`0.6073`

**Test（overall）**
- Accuracy：`0.7564`
- Recall：`0.8092`
- Specificity：`0.6484`
- Precision：`0.8249`
- F1：`0.8170`
- Balanced Acc：`0.7288`
- 混淆矩阵：TP=`212` TN=`83` FP=`45` FN=`50`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`39`
- 验证最优轮次（best_epoch）：`21`
- Val 选模分数（Balanced Acc）：`0.6582`
- Val F1（最优 checkpoint 时，附报）：`0.7074`
- Val loss（最优时）：`0.6301`

**Test（overall）**
- Accuracy：`0.4973`
- Recall：`0.4137`
- Specificity：`0.6723`
- Precision：`0.7254`
- F1：`0.5269`
- Balanced Acc：`0.5430`
- 混淆矩阵：TP=`103` TN=`80` FP=`39` FN=`146`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.7506`
- Val F1（最优 checkpoint 时，附报）：`0.8151`
- Val loss（最优时）：`0.6078`

**Test（overall）**
- Accuracy：`0.8024`
- Recall：`0.8448`
- Specificity：`0.7059`
- Precision：`0.8673`
- F1：`0.8559`
- Balanced Acc：`0.7754`
- 混淆矩阵：TP=`98` TN=`36` FP=`15` FN=`18`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.7347 ± 0.0543`
- Val F1（附报）：`0.7871 ± 0.0408`
- Test Acc：`0.7130 ± 0.1101`
- Test Spec：`0.6782 ± 0.0998`
- Test Rec：`0.7293 ± 0.1617`
- Test Precision：`0.8215 ± 0.0627`
- Test F1：`0.7651 ± 0.1202`
- Test BalAcc：`0.7037 ± 0.0890`

- 结束：`2026-08-02T16:24:06`
