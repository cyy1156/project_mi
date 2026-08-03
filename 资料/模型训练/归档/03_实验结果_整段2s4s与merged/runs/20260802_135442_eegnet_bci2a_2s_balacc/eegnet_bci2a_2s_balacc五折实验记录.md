# 被试独立五折实验记录（20260802_135442 / eegnet_bci2a_2s_balacc）

- 开始：`2026-08-02T13:54:42`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，tag=`bci2a_2s`）
- 切窗说明：任务 Cue+2~4s；静息下一 Cue 前 2s；500@250Hz
- model：`eegnet_bci2a_2s_balacc`（EEGNet 普通 CE；**仅 Task**；无加权/无采样平衡/无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`500`
- 样本：N=`1719` Rest=`556` Task=`1163`
- 划分：被试独立五折（非 LOSO、非混合库）
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_bci2a_2s_balacc\bci2a_2s\run_20260802_135442`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`109`
- 验证最优轮次（best_epoch）：`91`
- Val 选模分数（Balanced Acc）：`0.7503`
- Val F1（最优 checkpoint 时，附报）：`0.8131`
- Val loss（最优时）：`0.5158`

**Test（overall）**
- Accuracy：`0.7222`
- Recall：`0.8614`
- Specificity：`0.4341`
- Precision：`0.7591`
- F1：`0.8070`
- Balanced Acc：`0.6478`
- 混淆矩阵：TP=`230` TN=`56` FP=`73` FN=`37`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val 选模分数（Balanced Acc）：`0.5082`
- Val F1（最优 checkpoint 时，附报）：`0.8113`
- Val loss（最优时）：`0.6807`

**Test（overall）**
- Accuracy：`0.6683`
- Recall：`0.9888`
- Specificity：`0.0000`
- Precision：`0.6734`
- F1：`0.8012`
- Balanced Acc：`0.4944`
- 混淆矩阵：TP=`266` TN=`0` FP=`129` FN=`3`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5334`
- Val F1（最优 checkpoint 时，附报）：`0.7606`
- Val loss（最优时）：`0.6904`

**Test（overall）**
- Accuracy：`0.6385`
- Recall：`0.8359`
- Specificity：`0.2344`
- Precision：`0.6909`
- F1：`0.7565`
- Balanced Acc：`0.5351`
- 混淆矩阵：TP=`219` TN=`30` FP=`98` FN=`43`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`23`
- 验证最优轮次（best_epoch）：`5`
- Val 选模分数（Balanced Acc）：`0.5579`
- Val F1（最优 checkpoint 时，附报）：`0.7687`
- Val loss（最优时）：`0.6874`

**Test（overall）**
- Accuracy：`0.6359`
- Recall：`0.7952`
- Specificity：`0.3025`
- Precision：`0.7046`
- F1：`0.7472`
- Balanced Acc：`0.5489`
- 混淆矩阵：TP=`198` TN=`36` FP=`83` FN=`51`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`19`
- 验证最优轮次（best_epoch）：`1`
- Val 选模分数（Balanced Acc）：`0.5776`
- Val F1（最优 checkpoint 时，附报）：`0.7884`
- Val loss（最优时）：`0.6910`

**Test（overall）**
- Accuracy：`0.5868`
- Recall：`0.7845`
- Specificity：`0.1373`
- Precision：`0.6741`
- F1：`0.7251`
- Balanced Acc：`0.4609`
- 混淆矩阵：TP=`91` TN=`7` FP=`44` FN=`25`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.5855 ± 0.0857`
- Val F1（附报）：`0.7884 ± 0.0214`
- Test Acc：`0.6503 ± 0.0444`
- Test Spec：`0.2217 ± 0.1470`
- Test Rec：`0.8532 ± 0.0733`
- Test Precision：`0.7004 ± 0.0315`
- Test F1：`0.7674 ± 0.0317`
- Test BalAcc：`0.5374 ± 0.0633`

- 结束：`2026-08-02T13:56:15`
