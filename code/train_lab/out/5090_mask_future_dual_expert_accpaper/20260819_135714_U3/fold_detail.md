# U3 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260819_135714_U3`
- Test Acc_paper: `0.5678 ± 0.0195`
- Test BalAcc_maj: `0.5744 ± 0.0196`
- Test win F1: `0.5635 ± 0.0176`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5719`
- Val BalAcc_maj（附报）：`0.5781`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5491`
- BalAcc_maj：`0.5530`
- Acc_majority：`0.5530`
- F1-macro（众数）：`0.5537`
- Recall-macro：`0.5530`
- Recall idle/left/right：`0.5436` / `0.5700` / `0.5455`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5498` | F1m：`0.5504` | Acc：`0.5498`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    598    284    218
  true1    156    627    317
  true2    215    285    600
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10020   4907   3773
  true1   2788  10578   5334
  true2   3566   4891  10243
```

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5841`
- Val BalAcc_maj（附报）：`0.5911`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5891`
- BalAcc_maj：`0.5933`
- Acc_majority：`0.5933`
- F1-macro（众数）：`0.5933`
- Recall-macro：`0.5933`
- Recall idle/left/right：`0.5182` / `0.7000` / `0.5618`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5833` | F1m：`0.5828` | Acc：`0.5833`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    570    333    197
  true1     89    770    241
  true2    127    355    618
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9386   5711   3603
  true1   1696  12886   4118
  true2   2243   6004  10453
```

#### Fold 2

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5419`
- Val BalAcc_maj（附报）：`0.5467`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5418`
- BalAcc_maj：`0.5497`
- Acc_majority：`0.5497`
- F1-macro（众数）：`0.5477`
- Recall-macro：`0.5497`
- Recall idle/left/right：`0.5745` / `0.6209` / `0.4536`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5401` | F1m：`0.5380` | Acc：`0.5401`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    632    267    201
  true1    262    683    155
  true2    223    378    499
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10487   4606   3607
  true1   4534  11458   2708
  true2   3966   6381   8353
```

#### Fold 3

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.6033`
- Val BalAcc_maj（附报）：`0.6089`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5882`
- BalAcc_maj：`0.5961`
- Acc_majority：`0.5961`
- F1-macro（众数）：`0.5934`
- Recall-macro：`0.5961`
- Recall idle/left/right：`0.4809` / `0.5882` / `0.7191`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5846` | F1m：`0.5823` | Acc：`0.5846`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    529    290    281
  true1    125    647    328
  true2     91    218    791
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8862   4947   4891
  true1   2247  10946   5507
  true2   1670   4044  12986
```

#### Fold 4

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.5078`
- Val BalAcc_maj（附报）：`0.5133`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5707`
- BalAcc_maj：`0.5800`
- Acc_majority：`0.5800`
- F1-macro（众数）：`0.5771`
- Recall-macro：`0.5800`
- Recall idle/left/right：`0.6920` / `0.5710` / `0.4770`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5668` | F1m：`0.5639` | Acc：`0.5668`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    692    162    146
  true1    269    571    160
  true2    305    218    477
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11549   2866   2585
  true1   4774   9348   2878
  true2   5202   3787   8011
```
