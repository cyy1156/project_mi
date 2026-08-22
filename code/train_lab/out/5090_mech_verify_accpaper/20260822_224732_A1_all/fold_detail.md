### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5689`
- Val BalAcc_maj（附报）：`0.5744`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5512`
- BalAcc_maj：`0.5585`
- Acc_majority：`0.5585`
- F1-macro（众数）：`0.5580`
- Recall-macro：`0.5585`
- Recall idle/left/right：`0.5555` / `0.6118` / `0.5082`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5443` | F1m：`0.5441` | Acc：`0.5443`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    611    297    192
  true1    162    673    265
  true2    248    293    559
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10084   5240   3376
  true1   3026  11061   4613
  true2   4234   5074   9392
```

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5663`
- Val BalAcc_maj（附报）：`0.5715`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5924`
- BalAcc_maj：`0.5997`
- Acc_majority：`0.5997`
- F1-macro（众数）：`0.6005`
- Recall-macro：`0.5997`
- Recall idle/left/right：`0.5791` / `0.6182` / `0.6018`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5872` | F1m：`0.5878` | Acc：`0.5872`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    637    246    217
  true1    148    680    272
  true2    158    280    662
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10680   4232   3788
  true1   2729  11335   4636
  true2   3000   4772  10928
```

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5178`
- Val BalAcc_maj（附报）：`0.5215`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5345`
- BalAcc_maj：`0.5403`
- Acc_majority：`0.5403`
- F1-macro（众数）：`0.5365`
- Recall-macro：`0.5403`
- Recall idle/left/right：`0.6336` / `0.5755` / `0.4118`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5315` | F1m：`0.5277` | Acc：`0.5315`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    697    257    146
  true1    327    633    140
  true2    288    359    453
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11637   4380   2683
  true1   5698  10608   2394
  true2   5084   6045   7571
```

#### Fold 3

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5944`
- Val BalAcc_maj（附报）：`0.5978`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5842`
- BalAcc_maj：`0.5903`
- Acc_majority：`0.5903`
- F1-macro（众数）：`0.5878`
- Recall-macro：`0.5903`
- Recall idle/left/right：`0.5173` / `0.5482` / `0.7055`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5755` | F1m：`0.5733` | Acc：`0.5755`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    569    258    273
  true1    182    603    315
  true2    148    176    776
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9565   4310   4825
  true1   3317   9882   5501
  true2   2486   3376  12838
```

#### Fold 4

- stopped_epoch：`35` | best_epoch：`15`
- Val Acc_paper（早停）：`0.4981`
- Val BalAcc_maj（附报）：`0.5059`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5637`
- BalAcc_maj：`0.5680`
- Acc_majority：`0.5680`
- F1-macro（众数）：`0.5641`
- Recall-macro：`0.5680`
- Recall idle/left/right：`0.7120` / `0.5120` / `0.4800`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5603` | F1m：`0.5571` | Acc：`0.5603`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    712    131    157
  true1    304    512    184
  true2    321    199    480
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11788   2267   2945
  true1   5254   8531   3215
  true2   5427   3317   8256
```
