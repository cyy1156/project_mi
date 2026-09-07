# Exp38 · 误差去相关选池

方案：`资料/模型训练/38_旁路_官方主交卷_误差去相关选池_accpaper/`

```powershell
conda activate cyy
cd D:\MI\code\train_lab\src\step\5070_challenge_exp38_diversity_pool_accpaper
python run_exp38.py --stage all
```

D1：`eegtcnet` / `deep4`（替代 8ch-only dgcnn）+ `fbcsp_lda` / `riemann_tsc`  
D2：嵌套贪心 G* vs A0=nested-S0
