# U123 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260819_211232_U123`
- Test Acc_paper: `0.5677 ± 0.0204`
- Test BalAcc_maj: `0.5735 ± 0.0198`
- Test win F1: `0.5624 ± 0.0186`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`33` | best_epoch：`13`
- Val Acc_paper（早停）：`0.5707`
- Val BalAcc_maj（附报）：`0.5796`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5527`
- BalAcc_maj：`0.5573`
- Acc_majority：`0.5573`
- F1-macro（众数）：`0.5575`
- Recall-macro：`0.5573`
- Recall idle/left/right：`0.5636` / `0.5836` / `0.5245`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5497` | F1m：`0.5501` | Acc：`0.5497`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    620    280    200
  true1    163    642    295
  true2    235    288    577
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10403   4761   3536
  true1   3030  10566   5104
  true2   3881   4950   9869
```

#### Fold 1

- stopped_epoch：`43` | best_epoch：`23`
- Val Acc_paper（早停）：`0.5822`
- Val BalAcc_maj（附报）：`0.5896`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5773`
- BalAcc_maj：`0.5824`
- Acc_majority：`0.5824`
- F1-macro（众数）：`0.5821`
- Recall-macro：`0.5824`
- Recall idle/left/right：`0.5309` / `0.6709` / `0.5455`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5722` | F1m：`0.5719` | Acc：`0.5722`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    584    314    202
  true1    121    738    241
  true2    166    334    600
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9837   5241   3622
  true1   2420  12235   4045
  true2   2895   5774  10031
```

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5441`
- Val BalAcc_maj（附报）：`0.5481`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5379`
- BalAcc_maj：`0.5455`
- Acc_majority：`0.5455`
- F1-macro（众数）：`0.5446`
- Recall-macro：`0.5455`
- Recall idle/left/right：`0.5700` / `0.5909` / `0.4755`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5364` | F1m：`0.5356` | Acc：`0.5364`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    627    263    210
  true1    282    650    168
  true2    220    357    523
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10486   4470   3744
  true1   4925  10823   2952
  true2   3774   6141   8785
```

#### Fold 3

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.6078`
- Val BalAcc_maj（附报）：`0.6122`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5967`
- BalAcc_maj：`0.6012`
- Acc_majority：`0.6012`
- F1-macro（众数）：`0.5990`
- Recall-macro：`0.6012`
- Recall idle/left/right：`0.4809` / `0.6182` / `0.7045`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5920` | F1m：`0.5900` | Acc：`0.5920`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    529    289    282
  true1    113    680    307
  true2     86    239    775
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8938   4792   4970
  true1   2062  11373   5265
  true2   1596   4203  12901
```

#### Fold 4

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5093`
- Val BalAcc_maj（附报）：`0.5156`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5740`
- BalAcc_maj：`0.5813`
- Acc_majority：`0.5813`
- F1-macro（众数）：`0.5796`
- Recall-macro：`0.5813`
- Recall idle/left/right：`0.6840` / `0.5200` / `0.5400`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5656` | F1m：`0.5644` | Acc：`0.5656`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    684    144    172
  true1    292    520    188
  true2    284    176    540
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11142   2736   3122
  true1   4991   8707   3302
  true2   4907   3094   8999
```
