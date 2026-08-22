### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5715`
- Val BalAcc_maj（附报）：`0.5748`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5636`
- BalAcc_maj：`0.5694`
- Acc_majority：`0.5694`
- F1-macro（众数）：`0.5702`
- Recall-macro：`0.5694`
- Recall idle/left/right：`0.5518` / `0.5573` / `0.5991`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5565` | F1m：`0.5573` | Acc：`0.5565`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    607    259    234
  true1    145    613    342
  true2    189    252    659
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10134   4533   4033
  true1   2664  10102   5934
  true2   3226   4488  10986
```

#### Fold 1

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.5789`
- Val BalAcc_maj（附报）：`0.5867`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5927`
- BalAcc_maj：`0.5976`
- Acc_majority：`0.5976`
- F1-macro（众数）：`0.5979`
- Recall-macro：`0.5976`
- Recall idle/left/right：`0.5191` / `0.7027` / `0.5709`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5906` | F1m：`0.5907` | Acc：`0.5906`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    571    327    202
  true1     75    773    252
  true2    118    354    628
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9590   5397   3713
  true1   1432  12874   4394
  true2   2133   5896  10671
```

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5452`
- Val BalAcc_maj（附报）：`0.5522`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5397`
- BalAcc_maj：`0.5445`
- Acc_majority：`0.5445`
- F1-macro（众数）：`0.5446`
- Recall-macro：`0.5445`
- Recall idle/left/right：`0.5600` / `0.5636` / `0.5100`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5400` | F1m：`0.5401` | Acc：`0.5400`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    616    243    241
  true1    269    620    211
  true2    187    352    561
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10291   4235   4174
  true1   4730  10447   3523
  true2   3221   5925   9554
```

#### Fold 3

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.6107`
- Val BalAcc_maj（附报）：`0.6159`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6027`
- BalAcc_maj：`0.6070`
- Acc_majority：`0.6070`
- F1-macro（众数）：`0.6042`
- Recall-macro：`0.6070`
- Recall idle/left/right：`0.5000` / `0.5836` / `0.7373`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5916` | F1m：`0.5893` | Acc：`0.5916`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    550    276    274
  true1    130    642    328
  true2     94    195    811
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9199   4702   4799
  true1   2347  10693   5660
  true2   1748   3658  13294
```

#### Fold 4

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.5144`
- Val BalAcc_maj（附报）：`0.5200`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5737`
- BalAcc_maj：`0.5820`
- Acc_majority：`0.5820`
- F1-macro（众数）：`0.5808`
- Recall-macro：`0.5820`
- Recall idle/left/right：`0.6270` / `0.6120` / `0.5070`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5677` | F1m：`0.5668` | Acc：`0.5677`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    627    201    172
  true1    231    612    157
  true2    247    246    507
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10398   3433   3169
  true1   4023   9972   3005
  true2   4133   4286   8581
```
