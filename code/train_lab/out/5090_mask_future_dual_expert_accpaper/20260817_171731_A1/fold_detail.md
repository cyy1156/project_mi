# A1 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260817_171731_A1`
- Test Acc_paper: `0.5754 ± 0.0208`
- Test BalAcc_maj: `0.5825 ± 0.0208`
- Test win F1: `0.5694 ± 0.0206`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`53` | best_epoch：`33`
- Val Acc_paper（早停）：`0.5852`
- Val BalAcc_maj（附报）：`0.5889`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5503`
- BalAcc_maj：`0.5573`
- Acc_majority：`0.5573`
- F1-macro（众数）：`0.5572`
- Recall-macro：`0.5573`
- Recall idle/left/right：`0.5673` / `0.5936` / `0.5109`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5523` | F1m：`0.5522` | Acc：`0.5523`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    624    289    187
  true1    174    653    273
  true2    234    304    562
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10517   4963   3220
  true1   3102  11014   4584
  true2   3989   5259   9452
```

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5881`
- Val BalAcc_maj（附报）：`0.5952`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6039`
- BalAcc_maj：`0.6121`
- Acc_majority：`0.6121`
- F1-macro（众数）：`0.6123`
- Recall-macro：`0.6121`
- Recall idle/left/right：`0.6055` / `0.6755` / `0.5555`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5976` | F1m：`0.5976` | Acc：`0.5976`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    666    260    174
  true1    118    743    239
  true2    177    312    611
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11001   4498   3201
  true1   2282  12362   4056
  true2   3084   5454  10162
```

#### Fold 2

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5485`
- Val BalAcc_maj（附报）：`0.5548`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5591`
- BalAcc_maj：`0.5661`
- Acc_majority：`0.5661`
- F1-macro（众数）：`0.5626`
- Recall-macro：`0.5661`
- Recall idle/left/right：`0.6773` / `0.5645` / `0.4564`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5541` | F1m：`0.5510` | Acc：`0.5541`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    745    194    161
  true1    310    621    169
  true2    266    332    502
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12290   3427   2983
  true1   5370  10353   2977
  true2   4534   5725   8441
```

#### Fold 3

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.6152`
- Val BalAcc_maj（附报）：`0.6222`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5955`
- BalAcc_maj：`0.6009`
- Acc_majority：`0.6009`
- F1-macro（众数）：`0.6012`
- Recall-macro：`0.6009`
- Recall idle/left/right：`0.5964` / `0.5918` / `0.6145`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5909` | F1m：`0.5913` | Acc：`0.5909`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    656    262    182
  true1    205    651    244
  true2    181    243    676
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11033   4427   3240
  true1   3591  10840   4269
  true2   3073   4348  11279
```

#### Fold 4

- stopped_epoch：`44` | best_epoch：`24`
- Val Acc_paper（早停）：`0.5270`
- Val BalAcc_maj（附报）：`0.5363`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5683`
- BalAcc_maj：`0.5763`
- Acc_majority：`0.5763`
- F1-macro（众数）：`0.5696`
- Recall-macro：`0.5763`
- Recall idle/left/right：`0.7740` / `0.5200` / `0.4350`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5618` | F1m：`0.5550` | Acc：`0.5618`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    774    119    107
  true1    352    520    128
  true2    377    188    435
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  12804   2152   2044
  true1   6021   8670   2309
  true2   6499   3322   7179
```
