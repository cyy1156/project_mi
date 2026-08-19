# P0 · Acc_paper 五折

- run_dir: `F:\Cyy\MI\code\train_lab\out\5090_mask_future_dual_expert_accpaper\20260817_181226_P0`
- Test Acc_paper: `0.5699 ± 0.0213`
- Test BalAcc_maj: `0.5752 ± 0.0211`
- Test win F1: `0.5660 ± 0.0190`

### Three 分折明细

说明：早停与选模均为 **Val Acc_paper**；训练集 **batch balance**（inverse-freq）。

#### Fold 0

- stopped_epoch：`40` | best_epoch：`20`
- Val Acc_paper（早停）：`0.5759`
- Val BalAcc_maj（附报）：`0.5852`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5573`
- BalAcc_maj：`0.5624`
- Acc_majority：`0.5624`
- F1-macro（众数）：`0.5626`
- Recall-macro：`0.5624`
- Recall idle/left/right：`0.5673` / `0.5864` / `0.5336`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5503` | F1m：`0.5506` | Acc：`0.5503`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    624    269    207
  true1    185    645    270
  true2    225    288    587
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10423   4715   3562
  true1   3451  10439   4810
  true2   3794   4897  10009
```

#### Fold 1

- stopped_epoch：`28` | best_epoch：`8`
- Val Acc_paper（早停）：`0.5744`
- Val BalAcc_maj（附报）：`0.5807`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5918`
- BalAcc_maj：`0.5976`
- Acc_majority：`0.5976`
- F1-macro（众数）：`0.5982`
- Recall-macro：`0.5976`
- Recall idle/left/right：`0.5300` / `0.6882` / `0.5745`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5874` | F1m：`0.5877` | Acc：`0.5874`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    583    319    198
  true1     85    757    258
  true2    126    342    632
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9750   5345   3605
  true1   1787  12599   4314
  true2   2223   5875  10602
```

#### Fold 2

- stopped_epoch：`25` | best_epoch：`5`
- Val Acc_paper（早停）：`0.5459`
- Val BalAcc_maj（附报）：`0.5507`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5352`
- BalAcc_maj：`0.5409`
- Acc_majority：`0.5409`
- F1-macro（众数）：`0.5406`
- Recall-macro：`0.5409`
- Recall idle/left/right：`0.5727` / `0.5573` / `0.4927`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5384` | F1m：`0.5382` | Acc：`0.5384`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    630    247    223
  true1    294    613    193
  true2    219    339    542
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10601   4259   3840
  true1   4968  10341   3391
  true2   3743   5694   9263
```

#### Fold 3

- stopped_epoch：`48` | best_epoch：`28`
- Val Acc_paper（早停）：`0.6070`
- Val BalAcc_maj（附报）：`0.6093`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5897`
- BalAcc_maj：`0.5936`
- Acc_majority：`0.5936`
- F1-macro（众数）：`0.5924`
- Recall-macro：`0.5936`
- Recall idle/left/right：`0.5391` / `0.5491` / `0.6927`
- n_trials：`3300`

**Test 窗级（附报）**
- BalAcc：`0.5847` | F1m：`0.5837` | Acc：`0.5847`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    593    258    249
  true1    168    604    328
  true2    131    207    762
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0   9972   4375   4353
  true1   2932  10206   5562
  true2   2369   3707  12624
```

#### Fold 4

- stopped_epoch：`26` | best_epoch：`6`
- Val Acc_paper（早停）：`0.5111`
- Val BalAcc_maj（附报）：`0.5181`
- Val loss（最优时）：`None`

**Test 试次级**
- Acc_paper：`0.5757`
- BalAcc_maj：`0.5813`
- Acc_majority：`0.5813`
- F1-macro（众数）：`0.5805`
- Recall-macro：`0.5813`
- Recall idle/left/right：`0.6370` / `0.5850` / `0.5220`
- n_trials：`3000`

**Test 窗级（附报）**
- BalAcc：`0.5702` | F1m：`0.5697` | Acc：`0.5702`
- 混淆矩阵 试次级众数（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0    637    177    186
  true1    246    585    169
  true2    246    232    522
```
- 混淆矩阵 窗级（行=真实, 列=预测）：
```
         pred0  pred1  pred2
  true0  10409   3190   3401
  true1   4239   9739   3022
  true2   4128   3940   8932
```
