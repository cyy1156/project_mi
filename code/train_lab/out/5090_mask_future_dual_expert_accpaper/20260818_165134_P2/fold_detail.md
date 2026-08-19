# P2 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260818_165134_P2`
- Test Acc_paper: `0.5703 ± 0.0221`
- Test BalAcc_maj: `0.5761 ± 0.0231`
- Test win F1: `0.5661 ± 0.0188`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5781`
- Val BalAcc_maj（附报）：`0.5804`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5433`
- BalAcc_maj：`0.5467`
- Acc_majority：`0.5467`
- F1-macro（众数）：`0.5475`
- Recall-macro：`0.5467`
- Recall idle/left/right：`0.5127` / `0.5445` / `0.5827`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5457` | F1m：`0.5462` | Acc：`0.5457`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    564    272    264
  true1    139    599    362
  true2    186    273    641
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9570   4715   4415
  true1   2546   9998   6156
  true2   3124   4533  11043
```

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.5800`
- Val BalAcc_maj（附报）：`0.5863`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5964`
- BalAcc_maj：`0.6024`
- Acc_majority：`0.6024`
- F1-macro（众数）：`0.6032`
- Recall-macro：`0.6024`
- Recall idle/left/right：`0.5555` / `0.6755` / `0.5764`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5924` | F1m：`0.5928` | Acc：`0.5924`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    611    292    197
  true1     93    743    264
  true2    146    320    634
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10157   5052   3491
  true1   1843  12453   4404
  true2   2633   5442  10625
```

#### Fold 2

- stopped_epoch：`38` | best_epoch：`18`
- Val Acc_paper（早停）：`0.5396`
- Val BalAcc_maj（附报）：`0.5474`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5470`
- BalAcc_maj：`0.5518`
- Acc_majority：`0.5518`
- F1-macro（众数）：`0.5501`
- Recall-macro：`0.5518`
- Recall idle/left/right：`0.5645` / `0.6255` / `0.4655`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5464` | F1m：`0.5450` | Acc：`0.5464`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    621    262    217
  true1    254    688    158
  true2    222    366    512
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10367   4600   3733
  true1   4416  11495   2789
  true2   3759   6150   8791
```

#### Fold 3

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5993`
- Val BalAcc_maj（附报）：`0.6048`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5924`
- BalAcc_maj：`0.5985`
- Acc_majority：`0.5985`
- F1-macro（众数）：`0.5947`
- Recall-macro：`0.5985`
- Recall idle/left/right：`0.4664` / `0.5782` / `0.7509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5847` | F1m：`0.5809` | Acc：`0.5847`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    513    268    319
  true1    107    636    357
  true2     71    203    826
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8617   4645   5438
  true1   2101  10430   6169
  true2   1366   3579  13755
```

#### Fold 4

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5093`
- Val BalAcc_maj（附报）：`0.5167`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5723`
- BalAcc_maj：`0.5810`
- Acc_majority：`0.5810`
- F1-macro（众数）：`0.5798`
- Recall-macro：`0.5810`
- Recall idle/left/right：`0.6680` / `0.5380` / `0.5370`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5668` | F1m：`0.5656` | Acc：`0.5668`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    668    148    184
  true1    276    538    186
  true2    273    190    537
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11065   2704   3231
  true1   4803   8896   3301
  true2   4754   3301   8945
```
