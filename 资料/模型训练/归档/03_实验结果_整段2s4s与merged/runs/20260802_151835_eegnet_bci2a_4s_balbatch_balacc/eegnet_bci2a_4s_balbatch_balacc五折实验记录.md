# 被试独立五折实验记录（20260802_151835 / eegnet_bci2a_4s_balbatch_balacc）

- 开始：`2026-08-02T15:18:35`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_4s`（prefix=`bci2a`，tag=`bci2a_4s`）
- 切窗说明：任务 Cue+0~4s；静息下一 Cue 前 4s；1000@250Hz
- model：`eegnet_bci2a_4s_balbatch_balacc`（EEGNet 普通 CE；**仅 Task**；batch balance；无加权 CE / 无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`1000`
- 样本：N=`1719` Rest=`556` Task=`1163`
- 划分：被试独立五折（非 LOSO、非混合库）
- train sampler：类逆频 `WeightedRandomSampler`（batch balance，与特异度臂 B1 一致）
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_bci2a_4s_balbatch_balacc\bci2a_4s\run_20260802_151835`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`99`
- 验证最优轮次（best_epoch）：`81`
- Val 选模分数（Balanced Acc）：`0.7882`
- Val F1（最优 checkpoint 时，附报）：`0.7835`
- Val loss（最优时）：`0.5555`

**Test（overall）**
- Accuracy：`0.7222`
- Recall：`0.8202`
- Specificity：`0.5194`
- Precision：`0.7794`
- F1：`0.7993`
- Balanced Acc：`0.6698`
- 混淆矩阵：TP=`219` TN=`67` FP=`62` FN=`48`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`61`
- 验证最优轮次（best_epoch）：`43`
- Val 选模分数（Balanced Acc）：`0.7280`
- Val F1（最优 checkpoint 时，附报）：`0.8016`
- Val loss（最优时）：`0.5932`

**Test（overall）**
- Accuracy：`0.7588`
- Recall：`0.6989`
- Specificity：`0.8837`
- Precision：`0.9261`
- F1：`0.7966`
- Balanced Acc：`0.7913`
- 混淆矩阵：TP=`188` TN=`114` FP=`15` FN=`81`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`91`
- 验证最优轮次（best_epoch）：`73`
- Val 选模分数（Balanced Acc）：`0.7568`
- Val F1（最优 checkpoint 时，附报）：`0.8667`
- Val loss（最优时）：`0.4885`

**Test（overall）**
- Accuracy：`0.7795`
- Recall：`0.8740`
- Specificity：`0.5859`
- Precision：`0.8121`
- F1：`0.8419`
- Balanced Acc：`0.7300`
- 混淆矩阵：TP=`229` TN=`75` FP=`53` FN=`33`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`68`
- 验证最优轮次（best_epoch）：`50`
- Val 选模分数（Balanced Acc）：`0.6569`
- Val F1（最优 checkpoint 时，附报）：`0.7311`
- Val loss（最优时）：`0.6225`

**Test（overall）**
- Accuracy：`0.5489`
- Recall：`0.5141`
- Specificity：`0.6218`
- Precision：`0.7399`
- F1：`0.6066`
- Balanced Acc：`0.5680`
- 混淆矩阵：TP=`128` TN=`74` FP=`45` FN=`121`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`30`
- 验证最优轮次（best_epoch）：`12`
- Val 选模分数（Balanced Acc）：`0.7656`
- Val F1（最优 checkpoint 时，附报）：`0.8423`
- Val loss（最优时）：`0.6123`

**Test（overall）**
- Accuracy：`0.8144`
- Recall：`0.8793`
- Specificity：`0.6667`
- Precision：`0.8571`
- F1：`0.8681`
- Balanced Acc：`0.7730`
- 混淆矩阵：TP=`102` TN=`34` FP=`17` FN=`14`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.7391 ± 0.0454`
- Val F1（附报）：`0.8050 ± 0.0472`
- Test Acc：`0.7248 ± 0.0929`
- Test Spec：`0.6555 ± 0.1239`
- Test Rec：`0.7573 ± 0.1379`
- Test Precision：`0.8229 ± 0.0644`
- Test F1：`0.7825 ± 0.0919`
- Test BalAcc：`0.7064 ± 0.0809`

- 结束：`2026-08-02T15:23:26`
