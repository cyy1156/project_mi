# U13 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260819_172911_U13`
- Test Acc_paper: `0.5753 ± 0.0218`
- Test BalAcc_maj: `0.5814 ± 0.0225`
- Test win F1: `0.5689 ± 0.0198`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5741`
- Val BalAcc_maj（附报）：`0.5785`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5539`
- BalAcc_maj：`0.5591`
- Acc_majority：`0.5591`
- F1-macro（众数）：`0.5600`
- Recall-macro：`0.5591`
- Recall idle/left/right：`0.5455` / `0.5809` / `0.5509`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5511` | F1m：`0.5520` | Acc：`0.5511`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    600    270    230
  true1    140    639    321
  true2    202    292    606
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10023   4735   3942
  true1   2718  10567   5415
  true2   3355   5016  10329
```

#### Fold 1

- stopped_epoch：`31` | best_epoch：`11`
- Val Acc_paper（早停）：`0.5822`
- Val BalAcc_maj（附报）：`0.5893`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5979`
- BalAcc_maj：`0.6042`
- Acc_majority：`0.6042`
- F1-macro（众数）：`0.6041`
- Recall-macro：`0.6042`
- Recall idle/left/right：`0.5291` / `0.7155` / `0.5682`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5906` | F1m：`0.5903` | Acc：`0.5906`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    582    321    197
  true1     72    787    241
  true2    128    347    625
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9585   5556   3559
  true1   1468  13077   4155
  true2   2195   6033  10472
```

#### Fold 2

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5430`
- Val BalAcc_maj（附报）：`0.5515`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5452`
- BalAcc_maj：`0.5503`
- Acc_majority：`0.5503`
- F1-macro（众数）：`0.5497`
- Recall-macro：`0.5503`
- Recall idle/left/right：`0.5318` / `0.6136` / `0.5055`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5406` | F1m：`0.5403` | Acc：`0.5406`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    585    268    247
  true1    235    675    190
  true2    186    358    556
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9776   4717   4207
  true1   4307  11066   3327
  true2   3190   6024   9486
```

#### Fold 3

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.6000`
- Val BalAcc_maj（附报）：`0.6056`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5961`
- BalAcc_maj：`0.6030`
- Acc_majority：`0.6030`
- F1-macro（众数）：`0.6003`
- Recall-macro：`0.6030`
- Recall idle/left/right：`0.4836` / `0.6091` / `0.7164`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5907` | F1m：`0.5886` | Acc：`0.5907`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    532    302    266
  true1    121    670    309
  true2     95    217    788
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9071   5005   4624
  true1   2290  11017   5393
  true2   1774   3874  13052
```

#### Fold 4

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.5185`
- Val BalAcc_maj（附报）：`0.5248`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5837`
- BalAcc_maj：`0.5903`
- Acc_majority：`0.5903`
- F1-macro（众数）：`0.5879`
- Recall-macro：`0.5903`
- Recall idle/left/right：`0.6960` / `0.5730` / `0.5020`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5751` | F1m：`0.5732` | Acc：`0.5751`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    696    155    149
  true1    242    573    185
  true2    298    200    502
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11450   2773   2777
  true1   4365   9338   3297
  true2   5046   3411   8543
```
