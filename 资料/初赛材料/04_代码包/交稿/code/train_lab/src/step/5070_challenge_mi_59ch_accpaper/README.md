# Exp34 · 轨 A · 官方集 59ch · RTX 5070

> 方案 v0.4 · device=5070

## 流水线

```powershell
# 0) 预处理（已可全量）
cd D:\MI\code\preprocess_lab
python -m src.datasets.challenge_mi.batch_3s --mode 59

# 1) 四成员 LOSO
cd D:\MI\code\train_lab\src\step\5070_challenge_mi_59ch_accpaper
python baseline_shallow.py --max-folds 1     # 冒烟
python baseline_shallow.py
python baseline_shallow_b.py
python baseline_eegnet.py
python baseline_conformer.py

# 2) E1f-A59（Val 重拟合 T/w）
python fit_e1f_a59.py --auto-latest

# 3) 主 submission（S-ens）
python predict_e1f_submission.py --e1f-json ..\..\..\out\5070_challenge_mi_59ch_accpaper\e1f_a59\e1f_XXXX.json
```

或：`资料/模型训练/34_…/run_exp34.ps1 -Stage trainA`

## 超参
batch 32×4（conformer 16×8）· AMP · patience 20 · 6-fold LOSO

## 产物
- `out/5070_challenge_mi_59ch_accpaper/<model>_…/run_*/three/fold*/best_three.pt`
- `…/e1f_a59/e1f_*.json`
- `…/submissions/submission_exp34_e1f_a59_sens_*.csv` **主交卷**
