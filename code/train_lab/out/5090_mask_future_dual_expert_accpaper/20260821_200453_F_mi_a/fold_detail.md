# F_mi_a · scheme21

Test Acc_paper: `0.5765 ± 0.0268`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`46` | best_epoch：`26`
- Val Acc_paper（早停）：`0.5785`
- Val BalAcc_maj（附报）：`0.5789`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5636`
- BalAcc_maj：`0.5648`
- Acc_majority：`0.5648`
- F1-macro（众数）：`0.5642`
- Recall-macro：`0.5648`
- Recall idle/left/right：`0.6236` / `0.5000` / `0.5709`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5597` | F1m：`0.5587` | Acc：`0.5597`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    686    193    221
  true1    205    550    345
  true2    247    225    628
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   4777   1369   1554
  true1   1533   3793   2374
  true2   1768   1574   4358
```

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.6015`
- Val BalAcc_maj（附报）：`0.6015`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6103`
- BalAcc_maj：`0.6106`
- Acc_majority：`0.6106`
- F1-macro（众数）：`0.6094`
- Recall-macro：`0.6106`
- Recall idle/left/right：`0.6100` / `0.7064` / `0.5155`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6065` | F1m：`0.6052` | Acc：`0.6065`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    671    273    156
  true1    124    777    199
  true2    184    349    567
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   4688   1911   1101
  true1    897   5402   1401
  true2   1307   2473   3920
```

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5389`
- Val BalAcc_maj（附报）：`0.5396`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5370`
- BalAcc_maj：`0.5373`
- Acc_majority：`0.5373`
- F1-macro（众数）：`0.5298`
- Recall-macro：`0.5373`
- Recall idle/left/right：`0.6955` / `0.5436` / `0.3727`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5319` | F1m：`0.5243` | Acc：`0.5319`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    765    206    129
  true1    411    598     91
  true2    332    358    410
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   5307   1437    956
  true1   2892   4145    663
  true2   2325   2539   2836
```

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6207`
- Val BalAcc_maj（附报）：`0.6230`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6021`
- BalAcc_maj：`0.6042`
- Acc_majority：`0.6042`
- F1-macro（众数）：`0.6013`
- Recall-macro：`0.6042`
- Recall idle/left/right：`0.5718` / `0.5118` / `0.7291`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5985` | F1m：`0.5959` | Acc：`0.5985`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    629    215    256
  true1    208    563    329
  true2    131    167    802
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   4457   1452   1791
  true1   1490   3878   2332
  true2    946   1264   5490
```

#### Fold 4

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.5237`
- Val BalAcc_maj（附报）：`0.5241`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5693`
- BalAcc_maj：`0.5693`
- Acc_majority：`0.5693`
- F1-macro（众数）：`0.5689`
- Recall-macro：`0.5693`
- Recall idle/left/right：`0.5390` / `0.5480` / `0.6210`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5641` | F1m：`0.5638` | Acc：`0.5641`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    539    190    271
  true1    206    548    246
  true2    196    183    621
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   3751   1332   1917
  true1   1418   3800   1782
  true2   1377   1327   4296
```
