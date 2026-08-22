# J1_tok · scheme21

Test Acc_paper: `0.5733 ± 0.0223`
Test BalAcc_maj: `0.5782 ± 0.0228`
Test win F1: `0.5689 ± 0.0194`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`57` | best_epoch：`37`
- Val Acc_paper（早停）：`0.5711`
- Val BalAcc_maj（附报）：`0.5770`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5445`
- BalAcc_maj：`0.5482`
- Acc_majority：`0.5482`
- F1-macro（众数）：`0.5488`
- Recall-macro：`0.5482`
- Recall idle/left/right：`0.5027` / `0.5800` / `0.5618`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5482` | F1m：`0.5486` | Acc：`0.5482`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    553    306    241
  true1    132    638    330
  true2    186    296    618
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9352   5187   4161
  true1   2394  10798   5508
  true2   3132   4965  10603
```

#### Fold 1

- stopped_epoch：`45` | best_epoch：`25`
- Val Acc_paper（早停）：`0.6041`
- Val BalAcc_maj（附报）：`0.6089`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5967`
- BalAcc_maj：`0.6006`
- Acc_majority：`0.6006`
- F1-macro（众数）：`0.6010`
- Recall-macro：`0.6006`
- Recall idle/left/right：`0.5018` / `0.7164` / `0.5836`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5903` | F1m：`0.5901` | Acc：`0.5903`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    552    318    230
  true1     50    788    262
  true2     91    367    642
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9106   5531   4063
  true1   1072  13208   4420
  true2   1584   6315  10801
```

#### Fold 2

- stopped_epoch：`51` | best_epoch：`31`
- Val Acc_paper（早停）：`0.5470`
- Val BalAcc_maj（附报）：`0.5522`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5488`
- BalAcc_maj：`0.5533`
- Acc_majority：`0.5533`
- F1-macro（众数）：`0.5516`
- Recall-macro：`0.5533`
- Recall idle/left/right：`0.5800` / `0.6273` / `0.4527`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5452` | F1m：`0.5434` | Acc：`0.5452`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    638    276    186
  true1    252    690    158
  true2    196    406    498
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10566   4817   3317
  true1   4306  11655   2739
  true2   3430   6905   8365
```

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6111`
- Val BalAcc_maj（附报）：`0.6163`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5939`
- BalAcc_maj：`0.5994`
- Acc_majority：`0.5994`
- F1-macro（众数）：`0.5976`
- Recall-macro：`0.5994`
- Recall idle/left/right：`0.5582` / `0.5264` / `0.7136`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5884` | F1m：`0.5868` | Acc：`0.5884`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    614    232    254
  true1    179    579    342
  true2    117    198    785
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10215   3968   4517
  true1   3206   9784   5710
  true2   2048   3643  13009
```

#### Fold 4

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5100`
- Val BalAcc_maj（附报）：`0.5167`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5827`
- BalAcc_maj：`0.5897`
- Acc_majority：`0.5897`
- F1-macro（众数）：`0.5887`
- Recall-macro：`0.5897`
- Recall idle/left/right：`0.6530` / `0.5760` / `0.5400`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5762` | F1m：`0.5756` | Acc：`0.5762`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    653    167    180
  true1    225    576    199
  true2    226    234    540
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10626   2985   3389
  true1   3873   9636   3491
  true2   3898   3977   9125
```
