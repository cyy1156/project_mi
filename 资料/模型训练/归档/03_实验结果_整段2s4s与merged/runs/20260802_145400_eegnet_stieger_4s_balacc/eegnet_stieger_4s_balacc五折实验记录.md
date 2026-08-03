# 被试独立五折实验记录（20260802_145400 / eegnet_stieger_4s_balacc）

- 开始：`2026-08-02T14:54:00`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\stieger_4s`（prefix=`stieger`，tag=`stieger_4s`）
- 切窗说明：反馈段最后 4s（Task/Rest 同取法）；1000@250Hz
- model：`eegnet_stieger_4s_balacc`（EEGNet 普通 CE；**仅 Task**；无加权/无采样平衡/无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`1000`
- 样本：N=`6592` Rest=`1722` Task=`4870`
- 划分：被试独立五折（非 LOSO、非混合库）
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_stieger_4s_balacc\stieger_4s\run_20260802_145400`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`45`
- 验证最优轮次（best_epoch）：`27`
- Val 选模分数（Balanced Acc）：`0.5423`
- Val F1（最优 checkpoint 时，附报）：`0.8568`
- Val loss（最优时）：`0.5710`

**Test（overall）**
- Accuracy：`0.7190`
- Recall：`0.9361`
- Specificity：`0.1235`
- Precision：`0.7455`
- F1：`0.8300`
- Balanced Acc：`0.5298`
- 混淆矩阵：TP=`1040` TN=`50` FP=`355` FN=`71`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`86`
- 验证最优轮次（best_epoch）：`68`
- Val 选模分数（Balanced Acc）：`0.5156`
- Val F1（最优 checkpoint 时，附报）：`0.8096`
- Val loss（最优时）：`0.6239`

**Test（overall）**
- Accuracy：`0.8840`
- Recall：`0.9399`
- Specificity：`0.6849`
- Precision：`0.9140`
- F1：`0.9267`
- Balanced Acc：`0.8124`
- 混淆矩阵：TP=`797` TN=`163` FP=`75` FN=`51`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`34`
- 验证最优轮次（best_epoch）：`16`
- Val 选模分数（Balanced Acc）：`0.5148`
- Val F1（最优 checkpoint 时，附报）：`0.8011`
- Val loss（最优时）：`0.6736`

**Test（overall）**
- Accuracy：`0.7661`
- Recall：`0.9951`
- Specificity：`0.0464`
- Precision：`0.7663`
- F1：`0.8658`
- Balanced Acc：`0.5208`
- 混淆矩阵：TP=`1010` TN=`15` FP=`308` FN=`5`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`43`
- 验证最优轮次（best_epoch）：`25`
- Val 选模分数（Balanced Acc）：`0.5303`
- Val F1（最优 checkpoint 时，附报）：`0.8637`
- Val loss（最优时）：`0.5526`

**Test（overall）**
- Accuracy：`0.6757`
- Recall：`0.9709`
- Specificity：`0.0455`
- Precision：`0.6847`
- F1：`0.8030`
- Balanced Acc：`0.5082`
- 混淆矩阵：TP=`1001` TN=`22` FP=`461` FN=`30`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`47`
- 验证最优轮次（best_epoch）：`29`
- Val 选模分数（Balanced Acc）：`0.5224`
- Val F1（最优 checkpoint 时，附报）：`0.8002`
- Val loss（最优时）：`0.6490`

**Test（overall）**
- Accuracy：`0.5475`
- Recall：`0.5815`
- Specificity：`0.4396`
- Precision：`0.7668`
- F1：`0.6614`
- Balanced Acc：`0.5105`
- 混淆矩阵：TP=`503` TN=`120` FP=`153` FN=`362`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.5251 ± 0.0103`
- Val F1（附报）：`0.8263 ± 0.0280`
- Test Acc：`0.7184 ± 0.1102`
- Test Spec：`0.2680 ± 0.2540`
- Test Rec：`0.8847 ± 0.1531`
- Test Precision：`0.7755 ± 0.0755`
- Test F1：`0.8174 ± 0.0883`
- Test BalAcc：`0.5763 ± 0.1183`

- 结束：`2026-08-02T15:08:37`
