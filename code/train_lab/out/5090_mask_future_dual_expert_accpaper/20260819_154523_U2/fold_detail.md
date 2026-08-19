# U2 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260819_154523_U2`
- Test Acc_paper: `0.5709 ± 0.0236`
- Test BalAcc_maj: `0.5758 ± 0.0233`
- Test win F1: `0.5637 ± 0.0206`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5785`
- Val BalAcc_maj（附报）：`0.5837`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5479`
- BalAcc_maj：`0.5524`
- Acc_majority：`0.5524`
- F1-macro（众数）：`0.5526`
- Recall-macro：`0.5524`
- Recall idle/left/right：`0.5500` / `0.5173` / `0.5900`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5432` | F1m：`0.5432` | Acc：`0.5432`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    605    247    248
  true1    169    569    362
  true2    230    221    649
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10031   4212   4457
  true1   3078   9413   6209
  true2   3800   3871  11029
```

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5722`
- Val BalAcc_maj（附报）：`0.5781`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5927`
- BalAcc_maj：`0.5970`
- Acc_majority：`0.5970`
- F1-macro（众数）：`0.5979`
- Recall-macro：`0.5970`
- Recall idle/left/right：`0.5482` / `0.6555` / `0.5873`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5866` | F1m：`0.5871` | Acc：`0.5866`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    603    299    198
  true1    110    721    269
  true2    141    313    646
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9913   5182   3605
  true1   2051  12068   4581
  true2   2478   5294  10928
```

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5374`
- Val BalAcc_maj（附报）：`0.5452`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5388`
- BalAcc_maj：`0.5445`
- Acc_majority：`0.5445`
- F1-macro（众数）：`0.5431`
- Recall-macro：`0.5445`
- Recall idle/left/right：`0.6127` / `0.5591` / `0.4618`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5386` | F1m：`0.5373` | Acc：`0.5386`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    674    222    204
  true1    343    615    142
  true2    251    341    508
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11246   3951   3503
  true1   5772  10366   2562
  true2   4269   5825   8606
```

#### Fold 3

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.6030`
- Val BalAcc_maj（附报）：`0.6081`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5976`
- BalAcc_maj：`0.6018`
- Acc_majority：`0.6018`
- F1-macro（众数）：`0.5991`
- Recall-macro：`0.6018`
- Recall idle/left/right：`0.4800` / `0.6100` / `0.7155`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5872` | F1m：`0.5851` | Acc：`0.5872`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    528    287    285
  true1    119    671    310
  true2     92    221    787
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8926   4909   4865
  true1   2165  11001   5534
  true2   1648   4037  13015
```

#### Fold 4

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.4989`
- Val BalAcc_maj（附报）：`0.5037`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5773`
- BalAcc_maj：`0.5833`
- Acc_majority：`0.5833`
- F1-macro（众数）：`0.5803`
- Recall-macro：`0.5833`
- Recall idle/left/right：`0.7180` / `0.5140` / `0.5180`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5686` | F1m：`0.5655` | Acc：`0.5686`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    718    125    157
  true1    285    514    201
  true2    314    168    518
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11912   2246   2842
  true1   5076   8365   3559
  true2   5412   2866   8722
```
