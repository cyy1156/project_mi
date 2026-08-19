# B1 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260817_213801_B1`
- Test Acc_paper: `0.5611 ± 0.0205`
- Test BalAcc_maj: `0.5669 ± 0.0221`
- Test win F1: `0.5557 ± 0.0197`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5726`
- Val BalAcc_maj（附报）：`0.5763`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5467`
- BalAcc_maj：`0.5512`
- Acc_majority：`0.5512`
- F1-macro（众数）：`0.5513`
- Recall-macro：`0.5512`
- Recall idle/left/right：`0.5473` / `0.5091` / `0.5973`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5465` | F1m：`0.5466` | Acc：`0.5465`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    602    234    264
  true1    179    560    361
  true2    210    233    657
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10244   4048   4408
  true1   3204   9409   6087
  true2   3657   4037  11006
```

#### Fold 1

- stopped_epoch：`59` | best_epoch：`39`
- Val Acc_paper（早停）：`0.5752`
- Val BalAcc_maj（附报）：`0.5811`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5770`
- BalAcc_maj：`0.5845`
- Acc_majority：`0.5845`
- F1-macro（众数）：`0.5845`
- Recall-macro：`0.5845`
- Recall idle/left/right：`0.5227` / `0.6564` / `0.5745`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5740` | F1m：`0.5739` | Acc：`0.5740`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    575    292    233
  true1    113    722    265
  true2    160    308    632
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9606   4903   4191
  true1   2170  12028   4502
  true2   2718   5412  10570
```

#### Fold 2

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5300`
- Val BalAcc_maj（附报）：`0.5381`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5282`
- BalAcc_maj：`0.5315`
- Acc_majority：`0.5315`
- F1-macro（众数）：`0.5308`
- Recall-macro：`0.5315`
- Recall idle/left/right：`0.5073` / `0.5900` / `0.4973`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5231` | F1m：`0.5226` | Acc：`0.5231`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    558    267    275
  true1    260    649    191
  true2    205    348    547
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9376   4615   4709
  true1   4513  10800   3387
  true2   3455   6073   9172
```

#### Fold 3

- stopped_epoch：`42` | best_epoch：`22`
- Val Acc_paper（早停）：`0.6174`
- Val BalAcc_maj（附报）：`0.6226`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5824`
- BalAcc_maj：`0.5897`
- Acc_majority：`0.5897`
- F1-macro（众数）：`0.5882`
- Recall-macro：`0.5897`
- Recall idle/left/right：`0.5164` / `0.5718` / `0.6809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5772` | F1m：`0.5759` | Acc：`0.5772`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    568    300    232
  true1    185    629    286
  true2    138    213    749
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9450   4982   4268
  true1   3236  10577   4887
  true2   2467   3877  12356
```

#### Fold 4

- stopped_epoch：`49` | best_epoch：`29`
- Val Acc_paper（早停）：`0.5041`
- Val BalAcc_maj（附报）：`0.5137`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5713`
- BalAcc_maj：`0.5773`
- Acc_majority：`0.5773`
- F1-macro（众数）：`0.5728`
- Recall-macro：`0.5773`
- Recall idle/left/right：`0.7190` / `0.5720` / `0.4410`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5639` | F1m：`0.5595` | Acc：`0.5639`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    719    160    121
  true1    296    572    132
  true2    356    203    441
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11966   2799   2235
  true1   5243   9345   2412
  true2   5889   3663   7448
```
