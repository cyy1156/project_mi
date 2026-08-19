# U1 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260819_115104_U1`
- Test Acc_paper: `0.5683 ± 0.0218`
- Test BalAcc_maj: `0.5748 ± 0.0214`
- Test win F1: `0.5652 ± 0.0184`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`32` | best_epoch：`12`
- Val Acc_paper（早停）：`0.5726`
- Val BalAcc_maj（附报）：`0.5778`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5536`
- BalAcc_maj：`0.5600`
- Acc_majority：`0.5600`
- F1-macro（众数）：`0.5606`
- Recall-macro：`0.5600`
- Recall idle/left/right：`0.5445` / `0.5345` / `0.6009`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5529` | F1m：`0.5535` | Acc：`0.5529`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    599    262    239
  true1    157    588    355
  true2    188    251    661
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10049   4572   4079
  true1   2791   9866   6043
  true2   3143   4453  11104
```

#### Fold 1

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5752`
- Val BalAcc_maj（附报）：`0.5822`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5845`
- BalAcc_maj：`0.5903`
- Acc_majority：`0.5903`
- F1-macro（众数）：`0.5882`
- Recall-macro：`0.5903`
- Recall idle/left/right：`0.4745` / `0.7309` / `0.5655`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5795` | F1m：`0.5777` | Acc：`0.5795`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    522    358    220
  true1     67    804    229
  true2    112    366    622
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8771   5932   3997
  true1   1366  13246   4088
  true2   1954   6251  10495
```

#### Fold 2

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.5456`
- Val BalAcc_maj（附报）：`0.5489`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5352`
- BalAcc_maj：`0.5424`
- Acc_majority：`0.5424`
- F1-macro（众数）：`0.5408`
- Recall-macro：`0.5424`
- Recall idle/left/right：`0.5173` / `0.6364` / `0.4736`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5389` | F1m：`0.5376` | Acc：`0.5389`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    569    286    245
  true1    233    700    167
  true2    183    396    521
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9762   4884   4054
  true1   4104  11678   2918
  true2   3087   6819   8794
```

#### Fold 3

- stopped_epoch：`27` | best_epoch：`7`
- Val Acc_paper（早停）：`0.6011`
- Val BalAcc_maj（附报）：`0.6074`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5964`
- BalAcc_maj：`0.6024`
- Acc_majority：`0.6024`
- F1-macro（众数）：`0.6017`
- Recall-macro：`0.6024`
- Recall idle/left/right：`0.5282` / `0.5873` / `0.6918`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5913` | F1m：`0.5906` | Acc：`0.5913`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    581    265    254
  true1    147    646    307
  true2    102    237    761
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9775   4580   4345
  true1   2724  10797   5179
  true2   1914   4188  12598
```

#### Fold 4

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.5174`
- Val BalAcc_maj（附报）：`0.5248`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5717`
- BalAcc_maj：`0.5787`
- Acc_majority：`0.5787`
- F1-macro（众数）：`0.5769`
- Recall-macro：`0.5787`
- Recall idle/left/right：`0.6800` / `0.5360` / `0.5200`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5680` | F1m：`0.5667` | Acc：`0.5680`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    680    148    172
  true1    269    536    195
  true2    291    189    520
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11194   2624   3182
  true1   4608   8979   3413
  true2   5026   3177   8797
```
