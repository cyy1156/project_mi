# B10 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_124640_B10`
- Test Acc_paper: `0.5696 ± 0.0258`
- Test BalAcc_maj: `0.5756 ± 0.0252`
- Test win F1: `0.5653 ± 0.0230`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`52` | best_epoch：`32`
- Val Acc_paper（早停）：`0.5674`
- Val BalAcc_maj（附报）：`0.5726`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5452`
- BalAcc_maj：`0.5530`
- Acc_majority：`0.5530`
- F1-macro（众数）：`0.5535`
- Recall-macro：`0.5530`
- Recall idle/left/right：`0.5600` / `0.5282` / `0.5709`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5499` | F1m：`0.5503` | Acc：`0.5499`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    616    263    221
  true1    184    581    335
  true2    214    258    628
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10342   4523   3835
  true1   3242   9930   5528
  true2   3693   4431  10576
```

#### Fold 1

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5830`
- Val BalAcc_maj（附报）：`0.5889`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5927`
- BalAcc_maj：`0.5988`
- Acc_majority：`0.5988`
- F1-macro（众数）：`0.5981`
- Recall-macro：`0.5988`
- Recall idle/left/right：`0.5991` / `0.6736` / `0.5236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5889` | F1m：`0.5883` | Acc：`0.5889`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    659    264    177
  true1    140    741    219
  true2    194    330    576
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10980   4572   3148
  true1   2621  12359   3720
  true2   3317   5685   9698
```

#### Fold 2

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.5433`
- Val BalAcc_maj（附报）：`0.5519`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5330`
- BalAcc_maj：`0.5388`
- Acc_majority：`0.5388`
- F1-macro（众数）：`0.5350`
- Recall-macro：`0.5388`
- Recall idle/left/right：`0.5773` / `0.6309` / `0.4082`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5323` | F1m：`0.5288` | Acc：`0.5323`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    635    282    183
  true1    279    694    127
  true2    230    421    449
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10558   4986   3156
  true1   4845  11677   2178
  true2   3953   7119   7628
```

#### Fold 3

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.6178`
- Val BalAcc_maj（附报）：`0.6256`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5967`
- BalAcc_maj：`0.6015`
- Acc_majority：`0.6015`
- F1-macro（众数）：`0.6008`
- Recall-macro：`0.6015`
- Recall idle/left/right：`0.5673` / `0.5673` / `0.6700`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5886` | F1m：`0.5880` | Acc：`0.5886`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    624    245    231
  true1    194    624    282
  true2    158    205    737
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10457   4236   4007
  true1   3472  10364   4864
  true2   2705   3796  12199
```

#### Fold 4

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5293`
- Val BalAcc_maj（附报）：`0.5344`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5807`
- BalAcc_maj：`0.5857`
- Acc_majority：`0.5857`
- F1-macro（众数）：`0.5825`
- Recall-macro：`0.5857`
- Recall idle/left/right：`0.7050` / `0.5710` / `0.4810`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5736` | F1m：`0.5710` | Acc：`0.5736`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    705    151    144
  true1    273    571    156
  true2    305    214    481
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11607   2667   2726
  true1   4735   9528   2737
  true2   5162   3717   8121
```
