### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5730`
- Val BalAcc_maj（附报）：`0.5778`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5564`
- BalAcc_maj：`0.5645`
- Acc_majority：`0.5645`
- F1-macro（众数）：`0.5651`
- Recall-macro：`0.5645`
- Recall idle/left/right：`0.5418` / `0.5664` / `0.5855`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5483` | F1m：`0.5487` | Acc：`0.5483`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    596    259    245
  true1    159    623    318
  true2    198    258    644
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9787   4648   4265
  true1   2833  10130   5737
  true2   3379   4480  10841
```

#### Fold 1

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5807`
- Val BalAcc_maj（附报）：`0.5863`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5952`
- BalAcc_maj：`0.6042`
- Acc_majority：`0.6042`
- F1-macro（众数）：`0.6049`
- Recall-macro：`0.6042`
- Recall idle/left/right：`0.5564` / `0.6827` / `0.5736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5909` | F1m：`0.5915` | Acc：`0.5909`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    612    283    205
  true1     95    751    254
  true2    139    330    631
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10327   4736   3637
  true1   1897  12384   4419
  true2   2558   5705  10437
```

#### Fold 2

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5393`
- Val BalAcc_maj（附报）：`0.5441`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5579`
- BalAcc_maj：`0.5630`
- Acc_majority：`0.5630`
- F1-macro（众数）：`0.5598`
- Recall-macro：`0.5630`
- Recall idle/left/right：`0.6745` / `0.5527` / `0.4618`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5540` | F1m：`0.5512` | Acc：`0.5540`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    742    176    182
  true1    333    608    159
  true2    272    320    508
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12216   3376   3108
  true1   5481  10339   2880
  true2   4586   5591   8523
```

#### Fold 3

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.6156`
- Val BalAcc_maj（附报）：`0.6200`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5836`
- BalAcc_maj：`0.5882`
- Acc_majority：`0.5882`
- F1-macro（众数）：`0.5862`
- Recall-macro：`0.5882`
- Recall idle/left/right：`0.5000` / `0.5727` / `0.6918`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5812` | F1m：`0.5799` | Acc：`0.5812`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    550    287    263
  true1    154    630    316
  true2    127    212    761
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9404   4641   4655
  true1   2623  10596   5481
  true2   2182   3913  12605
```

#### Fold 4

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.5370`
- Val BalAcc_maj（附报）：`0.5459`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5773`
- BalAcc_maj：`0.5830`
- Acc_majority：`0.5830`
- F1-macro（众数）：`0.5760`
- Recall-macro：`0.5830`
- Recall idle/left/right：`0.7590` / `0.5710` / `0.4190`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5705` | F1m：`0.5640` | Acc：`0.5705`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    759    138    103
  true1    317    571    112
  true2    363    218    419
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12570   2480   1950
  true1   5470   9385   2145
  true2   6056   3804   7140
```
