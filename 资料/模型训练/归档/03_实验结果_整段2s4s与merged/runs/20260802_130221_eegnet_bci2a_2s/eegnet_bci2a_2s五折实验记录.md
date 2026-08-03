# 被试独立五折实验记录（20260802_130221 / eegnet_bci2a_2s）

- 开始：`2026-08-02T13:02:21`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，tag=`bci2a_2s`）
- 切窗说明：任务 Cue+2~4s；静息下一 Cue 前 2s；500@250Hz
- model：`eegnet_bci2a_2s`（EEGNet 普通 CE；**仅 Task**；无加权/无采样平衡/无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`500`
- 样本：N=`1719` Rest=`556` Task=`1163`
- 划分：被试独立五折（非 LOSO、非混合库）
- 早停：Val F1（与 baseline_eegnet 一致）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_bci2a_2s\bci2a_2s\run_20260802_130221`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`52`
- 验证最优轮次（best_epoch）：`34`
- Val F1（最优）：`0.8361`
- Val loss（最优时）：`0.5757`

**Test（overall）**
- Accuracy：`0.6970`
- Recall：`0.9588`
- Specificity：`0.1550`
- Precision：`0.7014`
- F1：`0.8101`
- Balanced Acc：`0.5569`
- 混淆矩阵：TP=`256` TN=`20` FP=`109` FN=`11`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.8113`
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

- 早停/结束轮次（stopped_epoch）：`22`
- 验证最优轮次（best_epoch）：`4`
- Val F1（最优）：`0.8113`
- Val loss（最优时）：`0.6789`

**Test（overall）**
- Accuracy：`0.6718`
- Recall：`0.9962`
- Specificity：`0.0078`
- Precision：`0.6727`
- F1：`0.8031`
- Balanced Acc：`0.5020`
- 混淆矩阵：TP=`261` TN=`1` FP=`127` FN=`1`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`33`
- Val F1（最优）：`0.8361`
- Val loss（最优时）：`0.5954`

**Test（overall）**
- Accuracy：`0.6902`
- Recall：`0.9036`
- Specificity：`0.2437`
- Precision：`0.7143`
- F1：`0.7979`
- Balanced Acc：`0.5737`
- 混淆矩阵：TP=`225` TN=`29` FP=`90` FN=`24`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`50`
- 验证最优轮次（best_epoch）：`32`
- Val F1（最优）：`0.8246`
- Val loss（最优时）：`0.5798`

**Test（overall）**
- Accuracy：`0.7545`
- Recall：`0.8707`
- Specificity：`0.4902`
- Precision：`0.7953`
- F1：`0.8313`
- Balanced Acc：`0.6804`
- 混淆矩阵：TP=`101` TN=`25` FP=`26` FN=`15`

## 最终结论（Test 五折均值）

- Val F1：`0.8239 ± 0.0111`
- Test Acc：`0.6964 ± 0.0310`
- Test Spec：`0.1793 ± 0.1805`
- Test Rec：`0.9436 ± 0.0489`
- Test Precision：`0.7114 ± 0.0449`
- Test F1：`0.8087 ± 0.0120`
- Test BalAcc：`0.5615 ± 0.0669`

- 结束：`2026-08-02T13:03:50`
