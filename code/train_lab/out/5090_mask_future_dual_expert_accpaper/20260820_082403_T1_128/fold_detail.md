# T1_128 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260820_082403_T1_128`
- Test Acc_paper: `0.7906 ± 0.0092`
- Test BalAcc_maj: `0.7906 ± 0.0092`
- Test win F1: `0.7829 ± 0.0077`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`30` | best_epoch：`10`
- Val Acc_paper（早停）：`0.7922`
- Val BalAcc_maj（附报）：`0.7922`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7821`
- BalAcc_maj：`0.7821`
- Acc_majority：`0.7821`
- F1-macro（众数）：`0.7821`
- Recall-macro：`0.7821`
- Recall idle/left/right：`1.0000` / `0.6655` / `0.6809`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7782` | F1m：`0.7782` | Acc：`0.7782`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    732    368
  true2      0    351    749
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  12417   6283
  true2      0   6159  12541
```

#### Fold 1

- stopped_epoch：`22` | best_epoch：`2`
- Val Acc_paper（早停）：`0.7881`
- Val BalAcc_maj（附报）：`0.7881`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7882`
- BalAcc_maj：`0.7882`
- Acc_majority：`0.7882`
- F1-macro（众数）：`0.7866`
- Recall-macro：`0.7882`
- Recall idle/left/right：`1.0000` / `0.7691` / `0.5955`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7866` | F1m：`0.7850` | Acc：`0.7866`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    846    254
  true2      0    445    655
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  14308   4392
  true2      0   7581  11119
```

#### Fold 2

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.7693`
- Val BalAcc_maj（附报）：`0.7693`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7803`
- BalAcc_maj：`0.7803`
- Acc_majority：`0.7803`
- F1-macro（众数）：`0.7749`
- Recall-macro：`0.7803`
- Recall idle/left/right：`1.0000` / `0.8255` / `0.5155`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7755` | F1m：`0.7706` | Acc：`0.7755`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    908    192
  true2      0    533    567
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  15145   3555
  true2      0   9040   9660
```

#### Fold 3

- stopped_epoch：`56` | best_epoch：`36`
- Val Acc_paper（早停）：`0.8033`
- Val BalAcc_maj（附报）：`0.8033`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.8039`
- BalAcc_maj：`0.8039`
- Acc_majority：`0.8039`
- F1-macro（众数）：`0.8035`
- Recall-macro：`0.8039`
- Recall idle/left/right：`0.9991` / `0.6545` / `0.7582`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7920` | F1m：`0.7915` | Acc：`0.7920`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1099      0      1
  true1      0    720    380
  true2      0    266    834
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18678      5     17
  true1      2  11909   6789
  true2      3   4855  13842
```

#### Fold 4

- stopped_epoch：`29` | best_epoch：`9`
- Val Acc_paper（早停）：`0.7607`
- Val BalAcc_maj（附报）：`0.7607`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7987`
- BalAcc_maj：`0.7987`
- Acc_majority：`0.7987`
- F1-macro（众数）：`0.7959`
- Recall-macro：`0.7987`
- Recall idle/left/right：`1.0000` / `0.8150` / `0.5810`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.7919` | F1m：`0.7894` | Acc：`0.7919`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1000      0      0
  true1      0    815    185
  true2      0    419    581
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  17000      0      0
  true1      0  13542   3458
  true2      0   7156   9844
```
