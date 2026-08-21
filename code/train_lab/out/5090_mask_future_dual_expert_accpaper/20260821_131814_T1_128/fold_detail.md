# T1_128 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260821_131814_T1_128`
- Test Acc_paper: `0.5735 ± 0.0237`
- Test BalAcc_maj: `0.5800 ± 0.0241`
- Test win F1: `0.5681 ± 0.0214`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5667`
- Val BalAcc_maj（附报）：`0.5707`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5555`
- BalAcc_maj：`0.5597`
- Acc_majority：`0.5597`
- F1-macro（众数）：`0.5596`
- Recall-macro：`0.5597`
- Recall idle/left/right：`0.4845` / `0.5345` / `0.6600`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5577` | F1m：`0.5571` | Acc：`0.5577`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    533    261    306
  true1    108    588    404
  true2    128    246    726
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8993   4513   5194
  true1   2007   9944   6749
  true2   2215   4136  12349
```

#### Fold 1

- stopped_epoch：`24` | best_epoch：`4`
- Val Acc_paper（早停）：`0.5800`
- Val BalAcc_maj（附报）：`0.5844`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5867`
- BalAcc_maj：`0.5933`
- Acc_majority：`0.5933`
- F1-macro（众数）：`0.5922`
- Recall-macro：`0.5933`
- Recall idle/left/right：`0.4927` / `0.7082` / `0.5791`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5810` | F1m：`0.5798` | Acc：`0.5810`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    542    334    224
  true1     74    779    247
  true2    123    340    637
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   8966   5711   4023
  true1   1527  12848   4325
  true2   2100   5821  10779
```

#### Fold 2

- stopped_epoch：`21` | best_epoch：`1`
- Val Acc_paper（早停）：`0.5481`
- Val BalAcc_maj（附报）：`0.5570`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5376`
- BalAcc_maj：`0.5452`
- Acc_majority：`0.5452`
- F1-macro（众数）：`0.5441`
- Recall-macro：`0.5452`
- Recall idle/left/right：`0.5009` / `0.6373` / `0.4973`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5334` | F1m：`0.5323` | Acc：`0.5334`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    551    327    222
  true1    230    701    169
  true2    173    380    547
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9112   5558   4030
  true1   4042  11636   3022
  true2   2949   6575   9176
```

#### Fold 3

- stopped_epoch：`23` | best_epoch：`3`
- Val Acc_paper（早停）：`0.5907`
- Val BalAcc_maj（附报）：`0.5978`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.6039`
- BalAcc_maj：`0.6115`
- Acc_majority：`0.6115`
- F1-macro（众数）：`0.6093`
- Recall-macro：`0.6115`
- Recall idle/left/right：`0.4991` / `0.6118` / `0.7236`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5961` | F1m：`0.5940` | Acc：`0.5961`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    549    306    245
  true1    115    673    312
  true2     95    209    796
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9143   5123   4434
  true1   2160  11174   5366
  true2   1801   3772  13127
```

#### Fold 4

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5119`
- Val BalAcc_maj（附报）：`0.5185`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5837`
- BalAcc_maj：`0.5903`
- Acc_majority：`0.5903`
- F1-macro（众数）：`0.5864`
- Recall-macro：`0.5903`
- Recall idle/left/right：`0.6970` / `0.6080` / `0.4660`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5802` | F1m：`0.5772` | Acc：`0.5802`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    697    170    133
  true1    230    608    162
  true2    280    254    466
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  11527   3073   2400
  true1   4091  10004   2905
  true2   4805   4138   8057
```
