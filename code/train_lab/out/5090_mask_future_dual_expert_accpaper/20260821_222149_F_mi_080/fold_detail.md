# F_mi_080 · scheme21

Test Acc_paper: `0.5684 ± 0.0261`
Test BalAcc_maj: `0.5701 ± 0.0263`
Test win F1: `0.5670 ± 0.0259`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5726`
- Val BalAcc_maj（附报）：`0.5737`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5658`
- BalAcc_maj：`0.5676`
- Acc_majority：`0.5676`
- F1-macro（众数）：`0.5679`
- Recall-macro：`0.5676`
- Recall idle/left/right：`0.5982` / `0.5536` / `0.5509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5596` | F1m：`0.5599` | Acc：`0.5596`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    658    238    204
  true1    181    609    310
  true2    224    270    606
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   5867   2162   1871
  true1   1690   5384   2826
  true2   2062   2470   5368
```

#### Fold 1

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5937`
- Val BalAcc_maj（附报）：`0.5956`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6018`
- BalAcc_maj：`0.6030`
- Acc_majority：`0.6030`
- F1-macro（众数）：`0.6024`
- Recall-macro：`0.6030`
- Recall idle/left/right：`0.5582` / `0.7100` / `0.5409`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.6014` | F1m：`0.6009` | Acc：`0.6014`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    614    292    194
  true1    104    781    215
  true2    153    352    595
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   5551   2605   1744
  true1    971   6978   1951
  true2   1405   3161   5334
```

#### Fold 2

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.5422`
- Val BalAcc_maj（附报）：`0.5448`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5585`
- BalAcc_maj：`0.5603`
- Acc_majority：`0.5603`
- F1-macro（众数）：`0.5574`
- Recall-macro：`0.5603`
- Recall idle/left/right：`0.6282` / `0.6064` / `0.4464`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5580` | F1m：`0.5556` | Acc：`0.5580`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    691    217    192
  true1    285    667    148
  true2    241    368    491
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   6179   2001   1720
  true1   2639   5930   1331
  true2   2124   3311   4465
```

#### Fold 3

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.6215`
- Val BalAcc_maj（附报）：`0.6230`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5894`
- BalAcc_maj：`0.5918`
- Acc_majority：`0.5918`
- F1-macro（众数）：`0.5918`
- Recall-macro：`0.5918`
- Recall idle/left/right：`0.5627` / `0.5782` / `0.6345`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5901` | F1m：`0.5902` | Acc：`0.5901`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    619    280    201
  true1    206    636    258
  true2    162    240    698
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   5594   2436   1870
  true1   1851   5750   2299
  true2   1527   2191   6182
```

#### Fold 4

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.5296`
- Val BalAcc_maj（附报）：`0.5319`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5267`
- BalAcc_maj：`0.5277`
- Acc_majority：`0.5277`
- F1-macro（众数）：`0.5258`
- Recall-macro：`0.5277`
- Recall idle/left/right：`0.4420` / `0.6100` / `0.5310`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5302` | F1m：`0.5286` | Acc：`0.5302`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    442    292    266
  true1    157    610    233
  true2    162    307    531
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   4048   2593   2359
  true1   1426   5492   2082
  true2   1447   2778   4775
```
