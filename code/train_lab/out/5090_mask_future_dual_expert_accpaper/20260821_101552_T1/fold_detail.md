# T1 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260821_101552_T1`
- Test Acc_paper: `0.5679 ± 0.0211`
- Test BalAcc_maj: `0.5733 ± 0.0212`
- Test win F1: `0.5640 ± 0.0199`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5652`
- Val BalAcc_maj（附报）：`0.5693`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5376`
- BalAcc_maj：`0.5415`
- Acc_majority：`0.5415`
- F1-macro（众数）：`0.5418`
- Recall-macro：`0.5415`
- Recall idle/left/right：`0.5282` / `0.5555` / `0.5409`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5377` | F1m：`0.5379` | Acc：`0.5377`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    581    285    234
  true1    180    611    309
  true2    240    265    595
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9659   5014   4027
  true1   3159  10319   5222
  true2   3870   4642  10188
```

#### Fold 1

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5756`
- Val BalAcc_maj（附报）：`0.5822`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5812`
- BalAcc_maj：`0.5873`
- Acc_majority：`0.5873`
- F1-macro（众数）：`0.5873`
- Recall-macro：`0.5873`
- Recall idle/left/right：`0.4936` / `0.6582` / `0.6100`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5823` | F1m：`0.5822` | Acc：`0.5823`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    543    279    278
  true1     78    724    298
  true2    122    307    671
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9189   4787   4724
  true1   1572  12112   5016
  true2   2082   5252  11366
```

#### Fold 2

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5507`
- Val BalAcc_maj（附报）：`0.5570`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5503`
- BalAcc_maj：`0.5567`
- Acc_majority：`0.5567`
- F1-macro（众数）：`0.5559`
- Recall-macro：`0.5567`
- Recall idle/left/right：`0.5118` / `0.6373` / `0.5209`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5430` | F1m：`0.5424` | Acc：`0.5430`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    563    289    248
  true1    180    701    219
  true2    176    351    573
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9306   5120   4274
  true1   3316  11507   3877
  true2   2991   6061   9648
```

#### Fold 3

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.6033`
- Val BalAcc_maj（附报）：`0.6100`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5955`
- BalAcc_maj：`0.6000`
- Acc_majority：`0.6000`
- F1-macro（众数）：`0.5961`
- Recall-macro：`0.6000`
- Recall idle/left/right：`0.4627` / `0.5973` / `0.7400`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5882` | F1m：`0.5843` | Acc：`0.5882`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    509    285    306
  true1    107    657    336
  true2     79    207    814
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8472   4940   5288
  true1   1948  10947   5805
  true2   1373   3748  13579
```

#### Fold 4

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.5085`
- Val BalAcc_maj（附报）：`0.5185`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5750`
- BalAcc_maj：`0.5810`
- Acc_majority：`0.5810`
- F1-macro（众数）：`0.5799`
- Recall-macro：`0.5810`
- Recall idle/left/right：`0.6510` / `0.5290` / `0.5630`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5737` | F1m：`0.5729` | Acc：`0.5737`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    651    159    190
  true1    253    529    218
  true2    238    199    563
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10797   2733   3470
  true1   4246   8983   3771
  true2   4088   3432   9480
```
