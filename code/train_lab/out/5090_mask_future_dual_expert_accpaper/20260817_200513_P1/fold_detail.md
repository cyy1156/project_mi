# P1 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260817_200513_P1`
- Test Acc_paper: `0.5735 ± 0.0220`
- Test BalAcc_maj: `0.5803 ± 0.0217`
- Test win F1: `0.5654 ± 0.0175`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`36` | best_epoch：`16`
- Val Acc_paper（早停）：`0.5696`
- Val BalAcc_maj（附报）：`0.5767`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5533`
- BalAcc_maj：`0.5609`
- Acc_majority：`0.5609`
- F1-macro（众数）：`0.5608`
- Recall-macro：`0.5609`
- Recall idle/left/right：`0.6055` / `0.5282` / `0.5491`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5541` | F1m：`0.5539` | Acc：`0.5541`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    666    217    217
  true1    197    581    322
  true2    245    251    604
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11125   3904   3671
  true1   3578   9700   5422
  true2   4187   4253  10260
```

#### Fold 1

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5785`
- Val BalAcc_maj（附报）：`0.5863`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5942`
- BalAcc_maj：`0.5991`
- Acc_majority：`0.5991`
- F1-macro（众数）：`0.5979`
- Recall-macro：`0.5991`
- Recall idle/left/right：`0.4927` / `0.7545` / `0.5500`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5825` | F1m：`0.5806` | Acc：`0.5825`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    542    377    181
  true1     50    830    220
  true2    100    395    605
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8834   6542   3324
  true1   1199  13660   3841
  true2   1918   6599  10183
```

#### Fold 2

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5400`
- Val BalAcc_maj（附报）：`0.5467`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5445`
- BalAcc_maj：`0.5509`
- Acc_majority：`0.5509`
- F1-macro（众数）：`0.5492`
- Recall-macro：`0.5509`
- Recall idle/left/right：`0.5809` / `0.6173` / `0.4545`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5400` | F1m：`0.5384` | Acc：`0.5400`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    639    273    188
  true1    265    679    156
  true2    215    385    500
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10665   4726   3309
  true1   4686  11238   2776
  true2   3776   6534   8390
```

#### Fold 3

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.6033`
- Val BalAcc_maj（附报）：`0.6067`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6006`
- BalAcc_maj：`0.6079`
- Acc_majority：`0.6079`
- F1-macro（众数）：`0.6052`
- Recall-macro：`0.6079`
- Recall idle/left/right：`0.5155` / `0.5627` / `0.7455`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5883` | F1m：`0.5861` | Acc：`0.5883`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    567    248    285
  true1    131    619    350
  true2    106    174    820
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9512   4253   4935
  true1   2510  10132   6058
  true2   2000   3341  13359
```

#### Fold 4

- stopped_epoch：`39` | best_epoch：`19`
- Val Acc_paper（早停）：`0.5096`
- Val BalAcc_maj（附报）：`0.5156`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5750`
- BalAcc_maj：`0.5827`
- Acc_majority：`0.5827`
- F1-macro（众数）：`0.5795`
- Recall-macro：`0.5827`
- Recall idle/left/right：`0.6800` / `0.6000` / `0.4680`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5707` | F1m：`0.5682` | Acc：`0.5707`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    680    174    146
  true1    248    600    152
  true2    293    239    468
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11338   3024   2638
  true1   4580   9742   2678
  true2   5035   3940   8025
```
