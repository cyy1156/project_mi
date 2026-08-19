# A2 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260817_191822_A2`
- Test Acc_paper: `0.5667 ± 0.0199`
- Test BalAcc_maj: `0.5728 ± 0.0208`
- Test win F1: `0.5621 ± 0.0178`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5696`
- Val BalAcc_maj（附报）：`0.5744`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5527`
- BalAcc_maj：`0.5579`
- Acc_majority：`0.5579`
- F1-macro（众数）：`0.5585`
- Recall-macro：`0.5579`
- Recall idle/left/right：`0.5655` / `0.5236` / `0.5845`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5522` | F1m：`0.5527` | Acc：`0.5522`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    622    233    245
  true1    161    576    363
  true2    213    244    643
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10384   4099   4217
  true1   2975   9694   6031
  true2   3542   4257  10901
```

#### Fold 1

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5711`
- Val BalAcc_maj（附报）：`0.5793`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5845`
- BalAcc_maj：`0.5909`
- Acc_majority：`0.5909`
- F1-macro（众数）：`0.5901`
- Recall-macro：`0.5909`
- Recall idle/left/right：`0.4664` / `0.7200` / `0.5864`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5800` | F1m：`0.5790` | Acc：`0.5800`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    513    362    225
  true1     52    792    256
  true2     76    379    645
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8605   6149   3946
  true1   1101  13132   4467
  true2   1535   6362  10803
```

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5407`
- Val BalAcc_maj（附报）：`0.5470`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5342`
- BalAcc_maj：`0.5391`
- Acc_majority：`0.5391`
- F1-macro（众数）：`0.5388`
- Recall-macro：`0.5391`
- Recall idle/left/right：`0.4927` / `0.5918` / `0.5327`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5319` | F1m：`0.5316` | Acc：`0.5319`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    542    310    248
  true1    260    651    189
  true2    174    340    586
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9183   5257   4260
  true1   4496  10946   3258
  true2   3060   5932   9708
```

#### Fold 3

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.6067`
- Val BalAcc_maj（附报）：`0.6104`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5830`
- BalAcc_maj：`0.5906`
- Acc_majority：`0.5906`
- F1-macro（众数）：`0.5887`
- Recall-macro：`0.5906`
- Recall idle/left/right：`0.5173` / `0.5564` / `0.6982`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5788` | F1m：`0.5771` | Acc：`0.5788`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    569    273    258
  true1    166    612    322
  true2    132    200    768
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9549   4537   4614
  true1   3004  10160   5536
  true2   2335   3601  12764
```

#### Fold 4

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5137`
- Val BalAcc_maj（附报）：`0.5215`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5790`
- BalAcc_maj：`0.5857`
- Acc_majority：`0.5857`
- F1-macro（众数）：`0.5846`
- Recall-macro：`0.5857`
- Recall idle/left/right：`0.6460` / `0.5930` / `0.5180`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5711` | F1m：`0.5701` | Acc：`0.5711`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    646    188    166
  true1    235    593    172
  true2    267    215    518
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10746   3228   3026
  true1   4219   9714   3067
  true2   4479   3857   8664
```
