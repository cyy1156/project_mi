# 实验 35 · 代码入口

> 方案：`资料/模型训练/35_旁路_官方vsOpenBMI_三分类排名不一致_融合重标定与骨干消融_accpaper/方案.md` **v0.2**  
> out：`code/train_lab/out/5070_challenge_rankflip_accpaper/`  
> 常驻排名表：`资料/模型训练/跨域三分类成员排名对照表.md`

## 一键全量（推荐）

```powershell
cd D:\MI\资料\模型训练\35_旁路_官方vsOpenBMI_三分类排名不一致_融合重标定与骨干消融_accpaper
.\run_exp35_full.ps1
# 或
python run_exp35_full.py

# 断点续跑 / 显式开 H / 不写 CSV
python run_exp35_full.py --resume
python run_exp35_full.py --with-h          # v0.2：轨 H 默认跳过
python run_exp35_full.py --skip-csv
python run_exp35_full.py --with-h --h-fold0-only
```

全量覆盖：**轨 R + F/M + S（三候选决赛 + 配对显著性 + 默认写 CSV）+ D**；**H 默认不开**。

日志：`资料/模型训练/35_…/logs/exp35_full_*.log`

## 分阶段（调试）

```powershell
cd D:\MI\code\train_lab\src\step\5070_challenge_rankflip_accpaper
conda activate cyy
python run_exp35.py --stage p0
python run_exp35.py --stage p0 --write-csv
python paired_sig_test.py --replay-json ..\..\..\out\5070_challenge_rankflip_accpaper\replay\replay_FM_latest.json
```

## 分步

| 脚本 | 作用 |
|------|------|
| `write_ranking_doc.py` | 轨 R → 常驻 + 本实验副本 |
| `export_member_probs.py` | Val/test 概率 dump |
| `replay_fusion_grid.py` | 轨 F/M/D 回放 |
| `make_submission_candidates.py` | 轨 S：S0+C_fuse+C_pool+C_conf 强制决赛 |
| `paired_sig_test.py` | 决赛 vs S0 Wilcoxon 附证（不换交卷） |
| `run_shallow_recipe_h.py` | 轨 H（可选） |
| `run_exp35.py` / `run_exp35_full.py` | 编排 |
