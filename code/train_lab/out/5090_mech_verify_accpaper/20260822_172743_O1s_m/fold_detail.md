### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.5574`
- Val BalAcc_maj（附报）：`0.5685`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5276`
- BalAcc_maj：`0.5391`
- Acc_majority：`0.5391`
- F1-macro（众数）：`0.5401`
- Recall-macro：`0.5391`
- Recall idle/left/right：`0.5045` / `0.5336` / `0.5791`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5306` | F1m：`0.5315` | Acc：`0.5306`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    555    291    254
  true1    137    587    376
  true2    183    280    637
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9366   4818   4516
  true1   2634   9712   6354
  true2   3048   4964  10688
```

#### Fold 1

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5789`
- Val BalAcc_maj（附报）：`0.5911`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5824`
- BalAcc_maj：`0.5885`
- Acc_majority：`0.5885`
- F1-macro（众数）：`0.5884`
- Recall-macro：`0.5885`
- Recall idle/left/right：`0.5455` / `0.7055` / `0.5145`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5747` | F1m：`0.5746` | Acc：`0.5747`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    600    314    186
  true1     77    776    247
  true2    149    385    566
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9898   5389   3413
  true1   1865  12653   4182
  true2   2576   6436   9688
```

#### Fold 2

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5281`
- Val BalAcc_maj（附报）：`0.5348`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5427`
- BalAcc_maj：`0.5561`
- Acc_majority：`0.5561`
- F1-macro（众数）：`0.5519`
- Recall-macro：`0.5561`
- Recall idle/left/right：`0.6091` / `0.6455` / `0.4136`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5404` | F1m：`0.5363` | Acc：`0.5404`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    670    284    146
  true1    239    710    151
  true2    230    415    455
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10931   5001   2768
  true1   4218  11863   2619
  true2   3890   7286   7524
```

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6022`
- Val BalAcc_maj（附报）：`0.6133`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5833`
- BalAcc_maj：`0.5958`
- Acc_majority：`0.5958`
- F1-macro（众数）：`0.5946`
- Recall-macro：`0.5958`
- Recall idle/left/right：`0.5809` / `0.5209` / `0.6855`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5735` | F1m：`0.5724` | Acc：`0.5735`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    639    229    232
  true1    180    573    347
  true2    148    198    754
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10564   3968   4168
  true1   3484   9402   5814
  true2   2783   3712  12205
```

#### Fold 4

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5311`
- Val BalAcc_maj（附报）：`0.5415`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5713`
- BalAcc_maj：`0.5837`
- Acc_majority：`0.5837`
- F1-macro（众数）：`0.5816`
- Recall-macro：`0.5837`
- Recall idle/left/right：`0.6670` / `0.5910` / `0.4930`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5611` | F1m：`0.5589` | Acc：`0.5611`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    667    202    131
  true1    238    591    171
  true2    257    250    493
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10982   3375   2643
  true1   4429   9623   2948
  true2   4507   4482   8011
```
