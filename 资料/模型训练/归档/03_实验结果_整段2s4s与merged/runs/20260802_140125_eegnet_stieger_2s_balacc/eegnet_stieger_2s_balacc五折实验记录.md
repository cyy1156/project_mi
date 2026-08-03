# 被试独立五折实验记录（20260802_140125 / eegnet_stieger_2s_balacc）

- 开始：`2026-08-02T14:01:25`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\stieger_2s`（prefix=`stieger`，tag=`stieger_2s`）
- 切窗说明：反馈段最后 2s（Task/Rest 同取法）；500@250Hz
- model：`eegnet_stieger_2s_balacc`（EEGNet 普通 CE；**仅 Task**；无加权/无采样平衡/无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`500`
- 样本：N=`34337` Rest=`9775` Task=`24562`
- 划分：被试独立五折（非 LOSO、非混合库）
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_stieger_2s_balacc\stieger_2s\run_20260802_140125`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`57`
- 验证最优轮次（best_epoch）：`39`
- Val 选模分数（Balanced Acc）：`0.6343`
- Val F1（最优 checkpoint 时，附报）：`0.7649`
- Val loss（最优时）：`0.6353`

**Test（overall）**
- Accuracy：`0.6658`
- Recall：`0.8099`
- Specificity：`0.3060`
- Precision：`0.7445`
- F1：`0.7758`
- Balanced Acc：`0.5579`
- 混淆矩阵：TP=`3872` TN=`586` FP=`1329` FN=`909`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`56`
- 验证最优轮次（best_epoch）：`38`
- Val 选模分数（Balanced Acc）：`0.5437`
- Val F1（最优 checkpoint 时，附报）：`0.8065`
- Val loss（最优时）：`0.6186`

**Test（overall）**
- Accuracy：`0.6085`
- Recall：`0.6396`
- Specificity：`0.5281`
- Precision：`0.7781`
- F1：`0.7021`
- Balanced Acc：`0.5839`
- 混淆矩阵：TP=`2616` TN=`835` FP=`746` FN=`1474`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`46`
- 验证最优轮次（best_epoch）：`28`
- Val 选模分数（Balanced Acc）：`0.5776`
- Val F1（最优 checkpoint 时，附报）：`0.7964`
- Val loss（最优时）：`0.6395`

**Test（overall）**
- Accuracy：`0.7735`
- Recall：`0.9450`
- Specificity：`0.2564`
- Precision：`0.7931`
- F1：`0.8624`
- Balanced Acc：`0.6007`
- 混淆矩阵：TP=`4246` TN=`382` FP=`1108` FN=`247`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`51`
- 验证最优轮次（best_epoch）：`33`
- Val 选模分数（Balanced Acc）：`0.6047`
- Val F1（最优 checkpoint 时，附报）：`0.7755`
- Val loss（最优时）：`0.6280`

**Test（overall）**
- Accuracy：`0.6867`
- Recall：`0.9408`
- Specificity：`0.1143`
- Precision：`0.7053`
- F1：`0.8062`
- Balanced Acc：`0.5275`
- 混淆矩阵：TP=`5099` TN=`275` FP=`2131` FN=`321`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`132`
- 验证最优轮次（best_epoch）：`114`
- Val 选模分数（Balanced Acc）：`0.5688`
- Val F1（最优 checkpoint 时，附报）：`0.8043`
- Val loss（最优时）：`0.6073`

**Test（overall）**
- Accuracy：`0.6819`
- Recall：`0.9095`
- Specificity：`0.1301`
- Precision：`0.7171`
- F1：`0.8019`
- Balanced Acc：`0.5198`
- 混淆矩阵：TP=`5255` TN=`310` FP=`2073` FN=`523`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.5858 ± 0.0311`
- Val F1（附报）：`0.7895 ± 0.0165`
- Test Acc：`0.6833 ± 0.0530`
- Test Spec：`0.2670 ± 0.1496`
- Test Rec：`0.8490 ± 0.1155`
- Test Precision：`0.7476 ± 0.0338`
- Test F1：`0.7897 ± 0.0521`
- Test BalAcc：`0.5580 ± 0.0312`

- 结束：`2026-08-02T14:53:59`
