### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5789`
- Val BalAcc_maj（附报）：`0.5863`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5594`
- BalAcc_maj：`0.5658`
- Acc_majority：`0.5658`
- F1-macro（众数）：`0.5665`
- Recall-macro：`0.5658`
- Recall idle/left/right：`0.5836` / `0.5400` / `0.5736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5543` | F1m：`0.5549` | Acc：`0.5543`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    642    235    223
  true1    157    594    349
  true2    217    252    631
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10465   4213   4022
  true1   2917   9963   5820
  true2   3611   4420  10669
```

#### Fold 1

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5937`
- Val BalAcc_maj（附报）：`0.6011`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6042`
- BalAcc_maj：`0.6085`
- Acc_majority：`0.6085`
- F1-macro（众数）：`0.6083`
- Recall-macro：`0.6085`
- Recall idle/left/right：`0.6136` / `0.6773` / `0.5345`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6028` | F1m：`0.6026` | Acc：`0.6028`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    675    251    174
  true1    129    745    226
  true2    174    338    588
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11402   4210   3088
  true1   2399  12456   3845
  true2   3004   5738   9958
```

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5456`
- Val BalAcc_maj（附报）：`0.5496`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5385`
- BalAcc_maj：`0.5461`
- Acc_majority：`0.5461`
- F1-macro（众数）：`0.5421`
- Recall-macro：`0.5461`
- Recall idle/left/right：`0.6191` / `0.6082` / `0.4109`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5372` | F1m：`0.5340` | Acc：`0.5372`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    681    259    160
  true1    307    669    124
  true2    259    389    452
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11134   4590   2976
  true1   5268  11247   2185
  true2   4353   6591   7756
```

#### Fold 3

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6152`
- Val BalAcc_maj（附报）：`0.6215`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5994`
- BalAcc_maj：`0.6045`
- Acc_majority：`0.6045`
- F1-macro（众数）：`0.6044`
- Recall-macro：`0.6045`
- Recall idle/left/right：`0.5900` / `0.5673` / `0.6564`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5942` | F1m：`0.5941` | Acc：`0.5942`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    649    242    209
  true1    183    624    293
  true2    158    220    722
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10896   4124   3680
  true1   3310  10412   4978
  true2   2783   3889  12028
```

#### Fold 4

- stopped_epoch：`61` | best_epoch：`41`
- Val Acc_paper（早停）：`0.5300`
- Val BalAcc_maj（附报）：`0.5344`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5800`
- BalAcc_maj：`0.5850`
- Acc_majority：`0.5850`
- F1-macro（众数）：`0.5819`
- Recall-macro：`0.5850`
- Recall idle/left/right：`0.7140` / `0.5520` / `0.4890`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5750` | F1m：`0.5724` | Acc：`0.5750`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    714    137    149
  true1    282    552    166
  true2    316    195    489
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11829   2504   2667
  true1   5014   9161   2825
  true2   5355   3311   8334
```
