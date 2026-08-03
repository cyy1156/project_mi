# 被试独立五折实验记录（20260802_151609 / eegnet_bci2a_2s_balbatch_balacc）

- 开始：`2026-08-02T15:16:09`
- device：`cuda`
- data：`D:\cyy\MI\code\preprocess_lab\out\bci2a_2s`（prefix=`bci2a`，tag=`bci2a_2s`）
- 切窗说明：任务 Cue+2~4s；静息下一 Cue 前 2s；500@250Hz
- model：`eegnet_bci2a_2s_balbatch_balacc`（EEGNet 普通 CE；**仅 Task**；batch balance；无加权 CE / 无 Focal）
- 结构：F1=8, D=2, F2=16；n_times=`500`
- 样本：N=`1719` Rest=`556` Task=`1163`
- 划分：被试独立五折（非 LOSO、非混合库）
- train sampler：类逆频 `WeightedRandomSampler`（batch balance，与特异度臂 B1 一致）
- 早停/选模：Val **Balanced Acc**（附报 Val F1）
- shared hp：`{'data_tag': 'merged_2s', 'n_folds': 5, 'val_ratio': 0.2, 'seed': 42, 'max_epochs': 300, 'patience': 18, 'batch_train': 32, 'batch_eval': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'drop_prob': 0.5}`
- 权重：`D:\cyy\MI\code\train_lab\out\eegnet_single_dataset\eegnet_bci2a_2s_balbatch_balacc\bci2a_2s\run_20260802_151609`

---
### Task 各折明细

说明：`stopped_epoch` 为早停触发（或跑满）时的轮次；`best_epoch` 为验证集最优并保存权重的轮次（旧基线多为 Val F1；特异度臂多为 Balanced Acc）。

#### Fold 0

- 早停/结束轮次（stopped_epoch）：`97`
- 验证最优轮次（best_epoch）：`79`
- Val 选模分数（Balanced Acc）：`0.7520`
- Val F1（最优 checkpoint 时，附报）：`0.7407`
- Val loss（最优时）：`0.5873`

**Test（overall）**
- Accuracy：`0.6414`
- Recall：`0.6442`
- Specificity：`0.6357`
- Precision：`0.7854`
- F1：`0.7078`
- Balanced Acc：`0.6399`
- 混淆矩阵：TP=`172` TN=`82` FP=`47` FN=`95`

#### Fold 1

- 早停/结束轮次（stopped_epoch）：`68`
- 验证最优轮次（best_epoch）：`50`
- Val 选模分数（Balanced Acc）：`0.6935`
- Val F1（最优 checkpoint 时，附报）：`0.7489`
- Val loss（最优时）：`0.6612`

**Test（overall）**
- Accuracy：`0.6658`
- Recall：`0.5911`
- Specificity：`0.8217`
- Precision：`0.8736`
- F1：`0.7051`
- Balanced Acc：`0.7064`
- 混淆矩阵：TP=`159` TN=`106` FP=`23` FN=`110`

#### Fold 2

- 早停/结束轮次（stopped_epoch）：`66`
- 验证最优轮次（best_epoch）：`48`
- Val 选模分数（Balanced Acc）：`0.6922`
- Val F1（最优 checkpoint 时，附报）：`0.7705`
- Val loss（最优时）：`0.6229`

**Test（overall）**
- Accuracy：`0.6436`
- Recall：`0.7137`
- Specificity：`0.5000`
- Precision：`0.7450`
- F1：`0.7290`
- Balanced Acc：`0.6069`
- 混淆矩阵：TP=`187` TN=`64` FP=`64` FN=`75`

#### Fold 3

- 早停/结束轮次（stopped_epoch）：`40`
- 验证最优轮次（best_epoch）：`22`
- Val 选模分数（Balanced Acc）：`0.6910`
- Val F1（最优 checkpoint 时，附报）：`0.7200`
- Val loss（最优时）：`0.6581`

**Test（overall）**
- Accuracy：`0.4864`
- Recall：`0.4378`
- Specificity：`0.5882`
- Precision：`0.6899`
- F1：`0.5356`
- Balanced Acc：`0.5130`
- 混淆矩阵：TP=`109` TN=`70` FP=`49` FN=`140`

#### Fold 4

- 早停/结束轮次（stopped_epoch）：`41`
- 验证最优轮次（best_epoch）：`23`
- Val 选模分数（Balanced Acc）：`0.6982`
- Val F1（最优 checkpoint 时，附报）：`0.7826`
- Val loss（最优时）：`0.6312`

**Test（overall）**
- Accuracy：`0.7186`
- Recall：`0.6983`
- Specificity：`0.7647`
- Precision：`0.8710`
- F1：`0.7751`
- Balanced Acc：`0.7315`
- 混淆矩阵：TP=`81` TN=`39` FP=`12` FN=`35`

## 最终结论（Test 五折均值）

- Val BalAcc（选模）：`0.7054 ± 0.0235`
- Val F1（附报）：`0.7526 ± 0.0221`
- Test Acc：`0.6312 ± 0.0775`
- Test Spec：`0.6621 ± 0.1170`
- Test Rec：`0.6170 ± 0.0995`
- Test Precision：`0.7930 ± 0.0715`
- Test F1：`0.6905 ± 0.0814`
- Test BalAcc：`0.6395 ± 0.0775`

- 结束：`2026-08-02T15:18:35`
