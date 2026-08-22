### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5774`
- Val BalAcc_maj（附报）：`0.5822`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5515`
- BalAcc_maj：`0.5588`
- Acc_majority：`0.5588`
- F1-macro（众数）：`0.5589`
- Recall-macro：`0.5588`
- Recall idle/left/right：`0.5927` / `0.5509` / `0.5327`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5525` | F1m：`0.5526` | Acc：`0.5525`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    652    250    198
  true1    185    606    309
  true2    252    262    586
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10935   4347   3418
  true1   3328  10189   5183
  true2   4262   4565   9873
```

#### Fold 1

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.5874`
- Val BalAcc_maj（附报）：`0.5907`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5988`
- BalAcc_maj：`0.6052`
- Acc_majority：`0.6052`
- F1-macro（众数）：`0.6060`
- Recall-macro：`0.6052`
- Recall idle/left/right：`0.6009` / `0.6455` / `0.5691`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5980` | F1m：`0.5987` | Acc：`0.5980`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    661    258    181
  true1    129    710    261
  true2    168    306    626
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11052   4376   3272
  true1   2266  12029   4405
  true2   2923   5311  10466
```

#### Fold 2

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5419`
- Val BalAcc_maj（附报）：`0.5463`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5621`
- BalAcc_maj：`0.5685`
- Acc_majority：`0.5685`
- F1-macro（众数）：`0.5653`
- Recall-macro：`0.5685`
- Recall idle/left/right：`0.6345` / `0.6255` / `0.4455`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5589` | F1m：`0.5559` | Acc：`0.5589`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    698    240    162
  true1    267    688    145
  true2    252    358    490
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11624   4149   2927
  true1   4614  11476   2610
  true2   4331   6112   8257
```

#### Fold 3

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.6156`
- Val BalAcc_maj（附报）：`0.6230`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5985`
- BalAcc_maj：`0.6039`
- Acc_majority：`0.6039`
- F1-macro（众数）：`0.6028`
- Recall-macro：`0.6039`
- Recall idle/left/right：`0.5864` / `0.5391` / `0.6864`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5914` | F1m：`0.5903` | Acc：`0.5914`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    645    220    235
  true1    201    593    306
  true2    151    194    755
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10692   3890   4118
  true1   3485   9923   5292
  true2   2673   3466  12561
```

#### Fold 4

- stopped_epoch：`69` | best_epoch：`49`
- Val Acc_paper（早停）：`0.5204`
- Val BalAcc_maj（附报）：`0.5237`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5613`
- BalAcc_maj：`0.5667`
- Acc_majority：`0.5667`
- F1-macro（众数）：`0.5591`
- Recall-macro：`0.5667`
- Recall idle/left/right：`0.7790` / `0.4530` / `0.4680`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5563` | F1m：`0.5499` | Acc：`0.5563`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    779     93    128
  true1    364    453    183
  true2    382    150    468
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12750   1739   2511
  true1   6177   7707   3116
  true2   6368   2718   7914
```
