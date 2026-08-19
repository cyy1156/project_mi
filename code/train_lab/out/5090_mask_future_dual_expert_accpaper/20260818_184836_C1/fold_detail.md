# C1 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_184836_C1`
- Test Acc_paper: `0.5693 ± 0.0246`
- Test BalAcc_maj: `0.5755 ± 0.0248`
- Test win F1: `0.5650 ± 0.0233`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5696`
- Val BalAcc_maj（附报）：`0.5748`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5661`
- BalAcc_maj：`0.5718`
- Acc_majority：`0.5718`
- F1-macro（众数）：`0.5726`
- Recall-macro：`0.5718`
- Recall idle/left/right：`0.5573` / `0.5645` / `0.5936`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5601` | F1m：`0.5608` | Acc：`0.5601`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    613    256    231
  true1    148    621    331
  true2    194    253    653
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10347   4458   3895
  true1   2827  10243   5630
  true2   3339   4531  10830
```

#### Fold 1

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.5833`
- Val BalAcc_maj（附报）：`0.5911`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5918`
- BalAcc_maj：`0.5979`
- Acc_majority：`0.5979`
- F1-macro（众数）：`0.5981`
- Recall-macro：`0.5979`
- Recall idle/left/right：`0.5218` / `0.6964` / `0.5755`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5873` | F1m：`0.5870` | Acc：`0.5873`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    574    327    199
  true1     77    766    257
  true2    128    339    633
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9448   5624   3628
  true1   1648  12740   4312
  true2   2276   5662  10762
```

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5470`
- Val BalAcc_maj（附报）：`0.5530`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5264`
- BalAcc_maj：`0.5318`
- Acc_majority：`0.5318`
- F1-macro（众数）：`0.5305`
- Recall-macro：`0.5318`
- Recall idle/left/right：`0.4909` / `0.6227` / `0.4818`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5263` | F1m：`0.5249` | Acc：`0.5263`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    540    304    256
  true1    230    685    185
  true2    173    397    530
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9048   5324   4328
  true1   3988  11564   3148
  true2   3104   6680   8916
```

#### Fold 3

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.6130`
- Val BalAcc_maj（附报）：`0.6196`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5948`
- BalAcc_maj：`0.6012`
- Acc_majority：`0.6012`
- F1-macro（众数）：`0.6009`
- Recall-macro：`0.6012`
- Recall idle/left/right：`0.5336` / `0.5927` / `0.6773`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5899` | F1m：`0.5895` | Acc：`0.5899`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    587    293    220
  true1    147    652    301
  true2    119    236    745
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9785   4826   4089
  true1   2652  10960   5088
  true2   2111   4242  12347
```

#### Fold 4

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5056`
- Val BalAcc_maj（附报）：`0.5104`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5673`
- BalAcc_maj：`0.5747`
- Acc_majority：`0.5747`
- F1-macro（众数）：`0.5735`
- Recall-macro：`0.5747`
- Recall idle/left/right：`0.6490` / `0.5740` / `0.5010`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5641` | F1m：`0.5629` | Acc：`0.5641`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    649    170    181
  true1    264    574    162
  true2    293    206    501
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10910   3016   3074
  true1   4714   9386   2900
  true2   4977   3551   8472
```
