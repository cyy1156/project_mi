# 被试独立五折实验记录（20260802_120001 / eegnet_focal_balacc）

- 开始：`2026-08-02T12:00:01`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\merged_2s`（prefix=`merged`）
- model：`eegnet_focal_balacc`（脚本 baseline_eegnet_focal_balacc.py；**不改** baseline_eegnet.py）
- 实验序号：`F1` — Focal Loss γ=2.0, α_rest=0.75, α_task=0.25 + BalAcc 早停
- 仅跑 Task（无 Three）
- 验收：Spec≥0.40 且 Rec≥0.75 且 BalAcc≥0.65（F1/Acc 仅附报）
- 对照：同数据上 EEGNet A22（加权CE w0=2.2）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\baseline\eegnet_focal_balacc\merged_2s\run_20260802_120001`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`82`
- 验证最优轮次（best_epoch）：`64`
- Val 选模分数（Balanced Acc）：`0.5554`
- Val F1（最优 checkpoint 时，附报）：`0.3594`
- Val loss（最优时）：`0.0644`

**Test（overall）**
- Accuracy：`0.4478`
- Recall：`0.3243`
- Specificity：`0.7529`
- Precision：`0.7642`
- F1：`0.4554`
- Balanced Acc：`0.5386`
- 混淆矩阵：TP=`1637` TN=`1539` FP=`505` FN=`3411`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5253` F1=`0.5026` BalAcc=`0.6159`
- `stieger_only`：Acc=`0.4432` F1=`0.4527` BalAcc=`0.5336`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`40`
- 验证最优轮次（best_epoch）：`22`
- Val 选模分数（Balanced Acc）：`0.5761`
- Val F1（最优 checkpoint 时，附报）：`0.7062`
- Val loss（最优时）：`0.0696`

**Test（overall）**
- Accuracy：`0.4141`
- Recall：`0.2576`
- Specificity：`0.8129`
- Precision：`0.7782`
- F1：`0.3871`
- Balanced Acc：`0.5352`
- 混淆矩阵：TP=`1123` TN=`1390` FP=`320` FN=`3236`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.3442` F1=`0.0645` BalAcc=`0.5129`
- `stieger_only`：Acc=`0.4190` F1=`0.4034` BalAcc=`0.5353`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`49`
- 验证最优轮次（best_epoch）：`31`
- Val 选模分数（Balanced Acc）：`0.5439`
- Val F1（最优 checkpoint 时，附报）：`0.6799`
- Val loss（最优时）：`0.0684`

**Test（overall）**
- Accuracy：`0.7373`
- Recall：`0.8118`
- Specificity：`0.5185`
- Precision：`0.8321`
- F1：`0.8218`
- Balanced Acc：`0.6652`
- 混淆矩阵：TP=`3860` TN=`839` FP=`779` FN=`895`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.7282` F1=`0.8215` BalAcc=`0.6219`
- `stieger_only`：Acc=`0.7379` F1=`0.8218` BalAcc=`0.6705`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val 选模分数（Balanced Acc）：`0.6041`
- Val F1（最优 checkpoint 时，附报）：`0.5985`
- Val loss（最优时）：`0.0652`

**Test（overall）**
- Accuracy：`0.6017`
- Recall：`0.7199`
- Specificity：`0.3362`
- Precision：`0.7089`
- F1：`0.7143`
- Balanced Acc：`0.5281`
- 混淆矩阵：TP=`4081` TN=`849` FP=`1676` FN=`1588`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.5924` F1=`0.6822` BalAcc=`0.5628`
- `stieger_only`：Acc=`0.6021` F1=`0.7157` BalAcc=`0.5262`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`42`
- 验证最优轮次（best_epoch）：`24`
- Val 选模分数（Balanced Acc）：`0.5987`
- Val F1（最优 checkpoint 时，附报）：`0.5963`
- Val loss（最优时）：`0.0650`

**Test（overall）**
- Accuracy：`0.5879`
- Recall：`0.6155`
- Specificity：`0.5210`
- Precision：`0.7568`
- F1：`0.6789`
- Balanced Acc：`0.5682`
- 混淆矩阵：TP=`3628` TN=`1268` FP=`1166` FN=`2266`

**Test（按数据集前缀）**
- `bci2a_only`：Acc=`0.4072` F1=`0.2878` BalAcc=`0.5568`
- `stieger_only`：Acc=`0.5916` F1=`0.6840` BalAcc=`0.5682`


## 汇总

- Test Acc：0.5578 ± 0.1165
- Test Spec：0.5883 ± 0.1735
- Test Rec：0.5458 ± 0.2182
- Test BalAcc：0.5671 ± 0.0509
- Test F1：0.6115 ± 0.1638
- 过关：否

对照 EEGNet A22：Spec≈0.439 Rec≈0.697 BalAcc≈0.568 Acc≈0.621
