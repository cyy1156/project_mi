# C2c · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_230041_C2c`
- Test Acc_paper: `0.5654 ± 0.0242`
- Test BalAcc_maj: `0.5708 ± 0.0241`
- Test win F1: `0.5633 ± 0.0222`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5711`
- Val BalAcc_maj（附报）：`0.5770`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5476`
- BalAcc_maj：`0.5542`
- Acc_majority：`0.5542`
- F1-macro（众数）：`0.5543`
- Recall-macro：`0.5542`
- Recall idle/left/right：`0.5627` / `0.4955` / `0.6045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5522` | F1m：`0.5522` | Acc：`0.5522`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    619    214    267
  true1    167    545    388
  true2    222    213    665
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10382   3872   4446
  true1   3014   9348   6338
  true2   3640   3813  11247
```

#### Fold 1

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.5885`
- Val BalAcc_maj（附报）：`0.5930`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5873`
- BalAcc_maj：`0.5930`
- Acc_majority：`0.5930`
- F1-macro（众数）：`0.5917`
- Recall-macro：`0.5930`
- Recall idle/left/right：`0.5264` / `0.7255` / `0.5273`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5843` | F1m：`0.5831` | Acc：`0.5843`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    579    316    205
  true1     82    798    220
  true2    143    377    580
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9703   5382   3615
  true1   1714  13264   3722
  true2   2436   6453   9811
```

#### Fold 2

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5504`
- Val BalAcc_maj（附报）：`0.5570`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5282`
- BalAcc_maj：`0.5330`
- Acc_majority：`0.5330`
- F1-macro（众数）：`0.5328`
- Recall-macro：`0.5330`
- Recall idle/left/right：`0.5118` / `0.5827` / `0.5045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5274` | F1m：`0.5272` | Acc：`0.5274`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    563    277    260
  true1    268    641    191
  true2    185    360    555
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9635   4744   4321
  true1   4885  10704   3111
  true2   3261   6191   9248
```

#### Fold 3

- stopped_epoch：`55` | best_epoch：`35`
- Val Acc_paper（早停）：`0.6100`
- Val BalAcc_maj（附报）：`0.6167`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5918`
- BalAcc_maj：`0.5970`
- Acc_majority：`0.5970`
- F1-macro（众数）：`0.5962`
- Recall-macro：`0.5970`
- Recall idle/left/right：`0.5236` / `0.5964` / `0.6709`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5892` | F1m：`0.5886` | Acc：`0.5892`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    576    308    216
  true1    162    656    282
  true2    131    231    738
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9743   5076   3881
  true1   2908  11053   4739
  true2   2251   4190  12259
```

#### Fold 4

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5156`
- Val BalAcc_maj（附报）：`0.5207`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5723`
- BalAcc_maj：`0.5767`
- Acc_majority：`0.5767`
- F1-macro（众数）：`0.5750`
- Recall-macro：`0.5767`
- Recall idle/left/right：`0.6690` / `0.5320` / `0.5290`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5671` | F1m：`0.5656` | Acc：`0.5671`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    669    166    165
  true1    269    532    199
  true2    267    204    529
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11141   2902   2957
  true1   4594   8921   3485
  true2   4577   3561   8862
```
