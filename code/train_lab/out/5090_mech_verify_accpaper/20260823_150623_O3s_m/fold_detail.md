### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`58` | best_epoch：`38`
- Val Acc_paper（早停）：`0.5752`
- Val BalAcc_maj（附报）：`0.5774`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5627`
- BalAcc_maj：`0.5670`
- Acc_majority：`0.5670`
- F1-macro（众数）：`0.5661`
- Recall-macro：`0.5670`
- Recall idle/left/right：`0.6045` / `0.6027` / `0.4936`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5528` | F1m：`0.5523` | Acc：`0.5528`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    665    268    167
  true1    179    663    258
  true2    254    303    543
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10916   4778   3006
  true1   3202  10968   4530
  true2   4275   5297   9128
```

#### Fold 1

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5844`
- Val BalAcc_maj（附报）：`0.5900`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6070`
- BalAcc_maj：`0.6103`
- Acc_majority：`0.6103`
- F1-macro（众数）：`0.6113`
- Recall-macro：`0.6103`
- Recall idle/left/right：`0.6009` / `0.6482` / `0.5818`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6035` | F1m：`0.6043` | Acc：`0.6035`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    661    225    214
  true1    118    713    269
  true2    162    298    640
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11106   3898   3696
  true1   2262  11984   4454
  true2   2871   5062  10767
```

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5404`
- Val BalAcc_maj（附报）：`0.5430`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5273`
- BalAcc_maj：`0.5300`
- Acc_majority：`0.5300`
- F1-macro（众数）：`0.5244`
- Recall-macro：`0.5300`
- Recall idle/left/right：`0.6973` / `0.4827` / `0.4100`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5232` | F1m：`0.5181` | Acc：`0.5232`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    767    206    127
  true1    424    531    145
  true2    371    278    451
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12729   3611   2360
  true1   7177   8927   2596
  true2   6104   4900   7696
```

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6133`
- Val BalAcc_maj（附报）：`0.6196`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5973`
- BalAcc_maj：`0.6009`
- Acc_majority：`0.6009`
- F1-macro（众数）：`0.5991`
- Recall-macro：`0.6009`
- Recall idle/left/right：`0.5055` / `0.6109` / `0.6864`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5943` | F1m：`0.5925` | Acc：`0.5943`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    556    266    278
  true1    145    672    283
  true2    133    212    755
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9342   4694   4664
  true1   2438  11260   5002
  true2   2232   3731  12737
```

#### Fold 4

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5144`
- Val BalAcc_maj（附报）：`0.5193`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5597`
- BalAcc_maj：`0.5627`
- Acc_majority：`0.5627`
- F1-macro（众数）：`0.5585`
- Recall-macro：`0.5627`
- Recall idle/left/right：`0.7260` / `0.5190` / `0.4430`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5533` | F1m：`0.5495` | Acc：`0.5533`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    726    136    138
  true1    360    519    121
  true2    375    182    443
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12053   2376   2571
  true1   6167   8655   2178
  true2   6277   3211   7512
```
