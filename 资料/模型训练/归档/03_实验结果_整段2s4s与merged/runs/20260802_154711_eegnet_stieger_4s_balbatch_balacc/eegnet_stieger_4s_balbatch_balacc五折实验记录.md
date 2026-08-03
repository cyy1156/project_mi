# 被试独立五折实验记录（20260802_154711 / eegnet_stieger_4s_balbatch_balacc）

- 开始：`2026-08-02T15:47:11`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\stieger_4s`（prefix=`stieger`，tag=`stieger_4s`）
- 切窗说明：反馈段最后 4s（Task/Rest 同取法）；1000@250Hz
- model：`eegnet_stieger_4s_balbatch_balacc`（EEGNet 普通 CE；**仅 Task**；batch balance；无加权 CE / 无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`1000`
- 样本：N=`6592` Rest=`1722` Task=`4870`
- 划分：被试独立五折（非 LOSO、非混合库）
- train sampler：类逆频 `WeightedRandomSampler`（batch balance，与特异度臂 B1 一致）
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_stieger_4s_balbatch_balacc\stieger_4s\run_20260802_154711`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc）：`0.6059`
- Val F1（最优 checkpoint 时，附报）：`0.7948`
- Val loss（最优时）：`0.6414`

**Test（overall）**
- Accuracy：`0.5858`
- Recall：`0.6265`
- Specificity：`0.4741`
- Precision：`0.7657`
- F1：`0.6891`
- Balanced Acc：`0.5503`
- 混淆矩阵：TP=`696` TN=`192` FP=`213` FN=`415`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`28`
- 验证最优轮次（best_epoch）：`10`
- Val 选模分数（Balanced Acc）：`0.5271`
- Val F1（最优 checkpoint 时，附报）：`0.5970`
- Val loss（最优时）：`0.6922`

**Test（overall）**
- Accuracy：`0.6971`
- Recall：`0.6722`
- Specificity：`0.7857`
- Precision：`0.9179`
- F1：`0.7760`
- Balanced Acc：`0.7289`
- 混淆矩阵：TP=`570` TN=`187` FP=`51` FN=`278`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5422`
- Val F1（最优 checkpoint 时，附报）：`0.6808`
- Val loss（最优时）：`0.6829`

**Test（overall）**
- Accuracy：`0.7646`
- Recall：`0.9645`
- Specificity：`0.1362`
- Precision：`0.7782`
- F1：`0.8614`
- Balanced Acc：`0.5504`
- 混淆矩阵：TP=`979` TN=`44` FP=`279` FN=`36`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`32`
- 验证最优轮次（best_epoch）：`14`
- Val 选模分数（Balanced Acc）：`0.5984`
- Val F1（最优 checkpoint 时，附报）：`0.8473`
- Val loss（最优时）：`0.6254`

**Test（overall）**
- Accuracy：`0.5291`
- Recall：`0.5325`
- Specificity：`0.5217`
- Precision：`0.7038`
- F1：`0.6063`
- Balanced Acc：`0.5271`
- 混淆矩阵：TP=`549` TN=`252` FP=`231` FN=`482`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.5473`
- Val F1（最优 checkpoint 时，附报）：`0.5259`
- Val loss（最优时）：`0.7334`

**Test（overall）**
- Accuracy：`0.4525`
- Recall：`0.4012`
- Specificity：`0.6154`
- Precision：`0.7677`
- F1：`0.5270`
- Balanced Acc：`0.5083`
- 混淆矩阵：TP=`347` TN=`168` FP=`105` FN=`518`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.5642 ± 0.0318`
- Val F1（附报）：`0.6891 ± 0.1195`
- Test Acc：`0.6058 ± 0.1125`
- Test Spec：`0.5066 ± 0.2136`
- Test Rec：`0.6394 ± 0.1872`
- Test Precision：`0.7867 ± 0.0706`
- Test F1：`0.6920 ± 0.1186`
- Test BalAcc：`0.5730 ± 0.0796`

- 结束：`2026-08-02T15:56:03`
