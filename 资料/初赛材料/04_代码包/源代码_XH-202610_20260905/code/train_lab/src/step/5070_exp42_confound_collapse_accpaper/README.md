# Exp42 · confound collapse diagnosis

## One-shot runner

```powershell
conda activate cyy
cd D:\MI\code\train_lab\src\step\5070_exp42_confound_collapse_accpaper

# Full pipeline: Day0 (P0/B/C/D/E) + Day1 Arm A + registry
python run_all.py --stage all --workers 4

# Day0 only (CPU features + stats; hours)
python run_all.py --stage day0 --workers 4

# Day1 only (GPU overnight; ~hundreds of all4 FT jobs)
python run_all.py --stage day1 --dry-run          # print queue, no train
python run_all.py --stage day1                    # real FT (resume-safe)

# Smoke
python run_all.py --stage day1 --people syj0828 --arms A1,A2 --seeds 0 --max-jobs 2
```

## Stages

| `--stage` | Contents |
|-----------|----------|
| `day0` | P0 cohort, session features, Leave-Next parse, B/C/D stats, E control, registry |
| `day1` / `a` | Arm A jackknife A1–A5 × seeds via `run_e1f_all4_finetune` |
| `summary` | refresh E + merge A aggregate into registry |
| `all` | day0 → day1 → summary |
| `e` | Arm E only |

## Outputs

- `资料/模型训练/42_.../analysis_42/*.json`
- `资料/模型训练/42_.../总结/结果登记表.md`
- `code/train_lab/out/5070_exp42_confound_collapse/` (`A/`, `run_all_log.json`, `replay_42_summary.json`)

Arm A writes only under `out/.../A/`; never touches `models/current`. Re-run skips jobs that already have `exp42_metrics.json`.

## After FT: repair metrics (no retrain)

If `exp42_metrics.json` were marked error (old F5 key bug), rebuild from disk:

```powershell
python reparse_A_metrics.py                 # window smooth from meta/gate
python reparse_A_metrics.py --eval-f5       # also recompute trial-level mi_acc_f5
```
