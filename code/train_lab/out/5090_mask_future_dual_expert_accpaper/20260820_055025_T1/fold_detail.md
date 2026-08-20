# T1 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260820_055025_T1`
- Test Acc_paper: `0.7946 ± 0.0123`
- Test BalAcc_maj: `0.7946 ± 0.0123`
- Test win F1: `0.7887 ± 0.0121`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.7941`
- Val BalAcc_maj（附报）：`0.7941`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7812`
- BalAcc_maj：`0.7812`
- Acc_majority：`0.7812`
- F1-macro（众数）：`0.7810`
- Recall-macro：`0.7812`
- Recall idle/left/right：`1.0000` / `0.6436` / `0.7000`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7747` | F1m：`0.7745` | Acc：`0.7747`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    708    392
  true2      0    330    770
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  11868   6832
  true2      0   5807  12893
```

#### Fold 1

- stopped_epoch：`37` | best_epoch：`17`
- Val Acc_paper（早停）：`0.7878`
- Val BalAcc_maj（附报）：`0.7878`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7885`
- BalAcc_maj：`0.7885`
- Acc_majority：`0.7885`
- F1-macro（众数）：`0.7876`
- Recall-macro：`0.7885`
- Recall idle/left/right：`1.0000` / `0.7482` / `0.6173`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7835` | F1m：`0.7826` | Acc：`0.7835`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    823    277
  true2      0    421    679
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  13848   4852
  true2      0   7291  11409
```

#### Fold 2

- stopped_epoch：`41` | best_epoch：`21`
- Val Acc_paper（早停）：`0.7737`
- Val BalAcc_maj（附报）：`0.7737`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.7845`
- BalAcc_maj：`0.7845`
- Acc_majority：`0.7845`
- F1-macro（众数）：`0.7811`
- Recall-macro：`0.7845`
- Recall idle/left/right：`1.0000` / `0.8018` / `0.5518`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.7834` | F1m：`0.7802` | Acc：`0.7834`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    882    218
  true2      0    493    607
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  14868   3832
  true2      0   8322  10378
```

#### Fold 3

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.8000`
- Val BalAcc_maj（附报）：`0.8000`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.8091`
- BalAcc_maj：`0.8091`
- Acc_majority：`0.8091`
- F1-macro（众数）：`0.8088`
- Recall-macro：`0.8091`
- Recall idle/left/right：`1.0000` / `0.6745` / `0.7527`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.8057` | F1m：`0.8054` | Acc：`0.8057`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1100      0      0
  true1      0    742    358
  true2      0    272    828
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  18700      0      0
  true1      0  12542   6158
  true2      0   4742  13958
```

#### Fold 4

- stopped_epoch：`47` | best_epoch：`27`
- Val Acc_paper（早停）：`0.7641`
- Val BalAcc_maj（附报）：`0.7641`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.8097`
- BalAcc_maj：`0.8097`
- Acc_majority：`0.8097`
- F1-macro（众数）：`0.8092`
- Recall-macro：`0.8097`
- Recall idle/left/right：`1.0000` / `0.7640` / `0.6650`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.8010` | F1m：`0.8006` | Acc：`0.8010`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   1000      0      0
  true1      0    764    236
  true2      0    335    665
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  17000      0      0
  true1      0  12663   4337
  true2      0   5813  11187
```
