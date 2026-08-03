# 被试独立五折实验记录（20260802_165141 / eegnet_stieger_4s_wce2p2_balacc）

- 开始：`2026-08-02T16:51:41`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\stieger_4s`（prefix=`stieger`，tag=`stieger_4s`）
- 切窗说明：反馈段最后 4s（Task/Rest 同取法）；1000@250Hz
- model：`eegnet_stieger_4s_wce2p2_balacc`（EEGNet 加权 CE w0=2.2/w1=1.0；**仅 Task**；无 batch balance / 无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`1000`
- 样本：N=`6592` Rest=`1722` Task=`4870`
- 划分：被试独立五折（非 LOSO、非混合库）
- train sampler：无（普通 shuffle）
- 损失：固定加权 CE，静息 w0=`2.2`，任务 w1=`1.0`
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_stieger_4s_wce2p2_balacc\stieger_4s\run_20260802_165141`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5826`
- Val F1（最优 checkpoint 时，附报）：`0.8204`
- Val loss（最优时）：`0.6571`

**Test（overall）**
- Accuracy：`0.6517`
- Recall：`0.7660`
- Specificity：`0.3383`
- Precision：`0.7605`
- F1：`0.7632`
- Balanced Acc：`0.5521`
- 混淆矩阵：TP=`851` TN=`137` FP=`268` FN=`260`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc）：`0.5572`
- Val F1（最优 checkpoint 时，附报）：`0.7380`
- Val loss（最优时）：`0.6906`

**Test（overall）**
- Accuracy：`0.8840`
- Recall：`0.9304`
- Specificity：`0.7185`
- Precision：`0.9217`
- F1：`0.9261`
- Balanced Acc：`0.8245`
- 混淆矩阵：TP=`789` TN=`171` FP=`67` FN=`59`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`29`
- 验证最优轮次（best_epoch）：`11`
- Val 选模分数（Balanced Acc）：`0.5413`
- Val F1（最优 checkpoint 时，附报）：`0.7572`
- Val loss（最优时）：`0.7049`

**Test（overall）**
- Accuracy：`0.7616`
- Recall：`0.9823`
- Specificity：`0.0681`
- Precision：`0.7681`
- F1：`0.8621`
- Balanced Acc：`0.5252`
- 混淆矩阵：TP=`997` TN=`22` FP=`301` FN=`18`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc）：`0.5485`
- Val F1（最优 checkpoint 时，附报）：`0.8398`
- Val loss（最优时）：`0.6597`

**Test（overall）**
- Accuracy：`0.6446`
- Recall：`0.8788`
- Specificity：`0.1449`
- Precision：`0.6869`
- F1：`0.7711`
- Balanced Acc：`0.5118`
- 混淆矩阵：TP=`906` TN=`70` FP=`413` FN=`125`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`36`
- 验证最优轮次（best_epoch）：`18`
- Val 选模分数（Balanced Acc）：`0.5522`
- Val F1（最优 checkpoint 时，附报）：`0.6290`
- Val loss（最优时）：`0.6920`

**Test（overall）**
- Accuracy：`0.4974`
- Recall：`0.5064`
- Specificity：`0.4689`
- Precision：`0.7513`
- F1：`0.6050`
- Balanced Acc：`0.4876`
- 混淆矩阵：TP=`438` TN=`128` FP=`145` FN=`427`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.5564 ± 0.0141`
- Val F1（附报）：`0.7569 ± 0.0743`
- Test Acc：`0.6879 ± 0.1291`
- Test Spec：`0.3477 ± 0.2330`
- Test Rec：`0.8128 ± 0.1691`
- Test Precision：`0.7777 ± 0.0776`
- Test F1：`0.7855 ± 0.1086`
- Test BalAcc：`0.5802 ± 0.1239`

- 结束：`2026-08-02T17:03:00`
