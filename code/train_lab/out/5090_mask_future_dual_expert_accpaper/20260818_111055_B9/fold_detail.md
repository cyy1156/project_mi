# B9 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_111055_B9`
- Test Acc_paper: `0.5788 ± 0.0189`
- Test BalAcc_maj: `0.5851 ± 0.0184`
- Test win F1: `0.5712 ± 0.0180`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`34` | best_epoch：`14`
- Val Acc_paper（早停）：`0.5689`
- Val BalAcc_maj（附报）：`0.5744`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5564`
- BalAcc_maj：`0.5633`
- Acc_majority：`0.5633`
- F1-macro（众数）：`0.5639`
- Recall-macro：`0.5633`
- Recall idle/left/right：`0.5600` / `0.5555` / `0.5745`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5556` | F1m：`0.5562` | Acc：`0.5556`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    616    243    241
  true1    160    611    329
  true2    224    244    632
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10426   4351   3923
  true1   2890  10241   5569
  true2   3789   4409  10502
```

#### Fold 1

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5956`
- Val BalAcc_maj（附报）：`0.6026`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5948`
- BalAcc_maj：`0.5991`
- Acc_majority：`0.5991`
- F1-macro（众数）：`0.5985`
- Recall-macro：`0.5991`
- Recall idle/left/right：`0.5473` / `0.7191` / `0.5309`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5907` | F1m：`0.5900` | Acc：`0.5907`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    602    305    193
  true1     90    791    219
  true2    140    376    584
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10111   5164   3425
  true1   1738  13171   3791
  true2   2495   6351   9854
```

#### Fold 2

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5419`
- Val BalAcc_maj（附报）：`0.5470`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5582`
- BalAcc_maj：`0.5648`
- Acc_majority：`0.5648`
- F1-macro（众数）：`0.5614`
- Recall-macro：`0.5648`
- Recall idle/left/right：`0.6682` / `0.5764` / `0.4500`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5533` | F1m：`0.5502` | Acc：`0.5533`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    735    201    164
  true1    312    634    154
  true2    267    338    495
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12116   3550   3034
  true1   5286  10605   2809
  true2   4551   5832   8317
```

#### Fold 3

- stopped_epoch：`50` | best_epoch：`30`
- Val Acc_paper（早停）：`0.6122`
- Val BalAcc_maj（附报）：`0.6170`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6030`
- BalAcc_maj：`0.6097`
- Acc_majority：`0.6097`
- F1-macro（众数）：`0.6086`
- Recall-macro：`0.6097`
- Recall idle/left/right：`0.5882` / `0.5491` / `0.6918`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5958` | F1m：`0.5946` | Acc：`0.5958`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    647    222    231
  true1    190    604    306
  true2    155    184    761
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10627   3862   4211
  true1   3389  10058   5253
  true2   2579   3380  12741
```

#### Fold 4

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5185`
- Val BalAcc_maj（附报）：`0.5256`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5817`
- BalAcc_maj：`0.5887`
- Acc_majority：`0.5887`
- F1-macro（众数）：`0.5862`
- Recall-macro：`0.5887`
- Recall idle/left/right：`0.7180` / `0.5380` / `0.5100`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5675` | F1m：`0.5648` | Acc：`0.5675`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    718    131    151
  true1    303    538    159
  true2    315    175    510
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11821   2352   2827
  true1   5375   8782   2843
  true2   5443   3218   8339
```
