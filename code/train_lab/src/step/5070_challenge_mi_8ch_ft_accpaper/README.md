# Exp34 · 轨 B · 官方 8ch · OpenBMI 热启动 · 5070

## 数据
```powershell
cd D:\MI\code\preprocess_lab
python -m src.datasets.challenge_mi.batch_3s --mode 8
```

## 训练
```powershell
cd D:\MI\code\train_lab\src\step\5070_challenge_mi_8ch_ft_accpaper
python baseline_shallow.py --max-folds 1          # FT 冒烟
python baseline_shallow.py                        # FT 满折
python baseline_shallow.py --scratch              # B8-scratch 消融
python baseline_shallow_b.py
python baseline_eegnet.py
python baseline_conformer.py
```

- 热启动：`experiment_game/config/e1f_four_member.json` 的 three ckpt  
- 标签轴：OpenBMI (Rest,L,R) → 挑战杯 (L,R,Rest) 自动 permute 分类头  

## E1f + 对照 CSV
```powershell
python fit_e1f_b8.py --auto-latest --arm ft
python predict_e1f_submission.py --e1f-json path\to\e1f_ft_*.json
```
