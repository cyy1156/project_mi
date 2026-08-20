# T1_aux · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260820_071519_T1_aux`
- Test Acc_paper: `0.7930 ± 0.0124`
- Test BalAcc_maj: `0.7930 ± 0.0124`
- Test win F1: `0.7857 ± 0.0121`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`69` | best_epoch：`49`
- Val Acc_paper（早停）：`0.7930`
- Val BalAcc_maj（附报）：`0.7930`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7836`
- BalAcc_maj：`0.7836`
- Acc_majority：`0.7836`
- F1-macro（众数）：`0.7835`
- Recall-macro：`0.7836`
- Recall idle/left/right：`1.0000` / `0.6482` / `0.7027`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7756` | F1m：`0.7755` | Acc：`0.7756`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    713    387
  true2      0    327    773
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  11980   6720
  true2      0   5868  12832
```

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.7900`
- Val BalAcc_maj（附报）：`0.7900`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7912`
- BalAcc_maj：`0.7912`
- Acc_majority：`0.7912`
- F1-macro（众数）：`0.7903`
- Recall-macro：`0.7912`
- Recall idle/left/right：`1.0000` / `0.7527` / `0.6209`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7849` | F1m：`0.7841` | Acc：`0.7849`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    828    272
  true2      0    417    683
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  13814   4886
  true2      0   7181  11519
```

#### Fold 2

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.7685`
- Val BalAcc_maj（附报）：`0.7685`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7815`
- BalAcc_maj：`0.7815`
- Acc_majority：`0.7815`
- F1-macro（众数）：`0.7766`
- Recall-macro：`0.7815`
- Recall idle/left/right：`1.0000` / `0.8209` / `0.5236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7779` | F1m：`0.7737` | Acc：`0.7779`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    903    197
  true2      0    524    576
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  15003   3697
  true2      0   8763   9937
```

#### Fold 3

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.8022`
- Val BalAcc_maj（附报）：`0.8022`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.8164`
- BalAcc_maj：`0.8164`
- Acc_majority：`0.8164`
- F1-macro（众数）：`0.8159`
- Recall-macro：`0.8164`
- Recall idle/left/right：`1.0000` / `0.6745` / `0.7745`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.8079` | F1m：`0.8074` | Acc：`0.8079`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    742    358
  true2      0    248    852
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  12390   6310
  true2      0   4466  14234
```

#### Fold 4

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.7641`
- Val BalAcc_maj（附报）：`0.7641`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7923`
- BalAcc_maj：`0.7923`
- Acc_majority：`0.7923`
- F1-macro（众数）：`0.7914`
- Recall-macro：`0.7923`
- Recall idle/left/right：`1.0000` / `0.7540` / `0.6230`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.7888` | F1m：`0.7877` | Acc：`0.7888`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1000      0      0
  true1      0    754    246
  true2      0    377    623
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  17000      0      0
  true1      1  12841   4158
  true2      0   6613  10387
```
