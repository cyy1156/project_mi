### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.5600`
- Val BalAcc_maj（附报）：`0.5711`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5427`
- BalAcc_maj：`0.5561`
- Acc_majority：`0.5561`
- F1-macro（众数）：`0.5570`
- Recall-macro：`0.5561`
- Recall idle/left/right：`0.5318` / `0.5591` / `0.5773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5357` | F1m：`0.5365` | Acc：`0.5357`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    585    284    231
  true1    140    615    345
  true2    191    274    635
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9571   4926   4203
  true1   2829   9964   5907
  true2   3327   4853  10520
```

#### Fold 1

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5744`
- Val BalAcc_maj（附报）：`0.5919`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5936`
- BalAcc_maj：`0.6024`
- Acc_majority：`0.6024`
- F1-macro（众数）：`0.6031`
- Recall-macro：`0.6024`
- Recall idle/left/right：`0.5555` / `0.7100` / `0.5418`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5796` | F1m：`0.5796` | Acc：`0.5796`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    611    301    188
  true1     80    781    239
  true2    123    381    596
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10065   5193   3442
  true1   1809  12762   4129
  true2   2523   6487   9690
```

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5170`
- Val BalAcc_maj（附报）：`0.5259`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5412`
- BalAcc_maj：`0.5530`
- Acc_majority：`0.5530`
- F1-macro（众数）：`0.5488`
- Recall-macro：`0.5530`
- Recall idle/left/right：`0.6227` / `0.6236` / `0.4127`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5395` | F1m：`0.5358` | Acc：`0.5395`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    685    266    149
  true1    263    686    151
  true2    251    395    454
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11275   4580   2845
  true1   4736  11318   2646
  true2   4213   6814   7673
```

#### Fold 3

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.6041`
- Val BalAcc_maj（附报）：`0.6137`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5800`
- BalAcc_maj：`0.5915`
- Acc_majority：`0.5915`
- F1-macro（众数）：`0.5913`
- Recall-macro：`0.5915`
- Recall idle/left/right：`0.5791` / `0.5391` / `0.6564`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5761` | F1m：`0.5759` | Acc：`0.5761`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    637    247    216
  true1    188    593    319
  true2    143    235    722
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10677   4169   3854
  true1   3421   9858   5421
  true2   2749   4168  11783
```

#### Fold 4

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.5278`
- Val BalAcc_maj（附报）：`0.5430`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5527`
- BalAcc_maj：`0.5637`
- Acc_majority：`0.5637`
- F1-macro（众数）：`0.5602`
- Recall-macro：`0.5637`
- Recall idle/left/right：`0.6980` / `0.4800` / `0.5130`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5470` | F1m：`0.5440` | Acc：`0.5470`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    698    146    156
  true1    302    480    218
  true2    314    173    513
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11378   2591   3031
  true1   5316   8001   3683
  true2   5292   3191   8517
```
