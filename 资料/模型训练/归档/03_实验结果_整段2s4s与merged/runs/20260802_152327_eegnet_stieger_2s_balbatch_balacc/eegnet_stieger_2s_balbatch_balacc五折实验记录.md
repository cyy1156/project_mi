# 被试独立五折实验记录（20260802_152327 / eegnet_stieger_2s_balbatch_balacc）

- 开始：`2026-08-02T15:23:27`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\stieger_2s`（prefix=`stieger`，tag=`stieger_2s`）
- 切窗说明：反馈段最后 2s（Task/Rest 同取法）；500@250Hz
- model：`eegnet_stieger_2s_balbatch_balacc`（EEGNet 普通 CE；**仅 Task**；batch balance；无加权 CE / 无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`500`
- 样本：N=`34337` Rest=`9775` Task=`24562`
- 划分：被试独立五折（非 LOSO、非混合库）
- train sampler：类逆频 `WeightedRandomSampler`（batch balance，与特异度臂 B1 一致）
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_stieger_2s_balbatch_balacc\stieger_2s\run_20260802_152327`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`21`
- 验证最优轮次（best_epoch）：`3`
- Val 选模分数（Balanced Acc）：`0.6023`
- Val F1（最优 checkpoint 时，附报）：`0.6576`
- Val loss（最优时）：`0.6854`

**Test（overall）**
- Accuracy：`0.5747`
- Recall：`0.5926`
- Specificity：`0.5300`
- Precision：`0.7589`
- F1：`0.6655`
- Balanced Acc：`0.5613`
- 混淆矩阵：TP=`2833` TN=`1015` FP=`900` FN=`1948`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`38`
- 验证最优轮次（best_epoch）：`20`
- Val 选模分数（Balanced Acc）：`0.5800`
- Val F1（最优 checkpoint 时，附报）：`0.7664`
- Val loss（最优时）：`0.6469`

**Test（overall）**
- Accuracy：`0.4706`
- Recall：`0.3782`
- Specificity：`0.7097`
- Precision：`0.7712`
- F1：`0.5075`
- Balanced Acc：`0.5440`
- 混淆矩阵：TP=`1547` TN=`1122` FP=`459` FN=`2543`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5409`
- Val F1（最优 checkpoint 时，附报）：`0.5371`
- Val loss（最优时）：`0.6957`

**Test（overall）**
- Accuracy：`0.5761`
- Recall：`0.5226`
- Specificity：`0.7376`
- Precision：`0.8572`
- F1：`0.6493`
- Balanced Acc：`0.6301`
- 混淆矩阵：TP=`2348` TN=`1099` FP=`391` FN=`2145`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`24`
- 验证最优轮次（best_epoch）：`6`
- Val 选模分数（Balanced Acc）：`0.5990`
- Val F1（最优 checkpoint 时，附报）：`0.6171`
- Val loss（最优时）：`0.6920`

**Test（overall）**
- Accuracy：`0.6055`
- Recall：`0.7220`
- Specificity：`0.3433`
- Precision：`0.7124`
- F1：`0.7171`
- Balanced Acc：`0.5326`
- 混淆矩阵：TP=`3913` TN=`826` FP=`1580` FN=`1507`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`48`
- 验证最优轮次（best_epoch）：`30`
- Val 选模分数（Balanced Acc）：`0.6143`
- Val F1（最优 checkpoint 时，附报）：`0.7016`
- Val loss（最优时）：`0.6758`

**Test（overall）**
- Accuracy：`0.6293`
- Recall：`0.7269`
- Specificity：`0.3928`
- Precision：`0.7438`
- F1：`0.7352`
- Balanced Acc：`0.5598`
- 混淆矩阵：TP=`4200` TN=`936` FP=`1447` FN=`1578`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.5873 ± 0.0257`
- Val F1（附报）：`0.6559 ± 0.0774`
- Test Acc：`0.5713 ± 0.0542`
- Test Spec：`0.5427 ± 0.1602`
- Test Rec：`0.5884 ± 0.1308`
- Test Precision：`0.7687 ± 0.0485`
- Test F1：`0.6549 ± 0.0802`
- Test BalAcc：`0.5656 ± 0.0340`

- 结束：`2026-08-02T15:47:11`
