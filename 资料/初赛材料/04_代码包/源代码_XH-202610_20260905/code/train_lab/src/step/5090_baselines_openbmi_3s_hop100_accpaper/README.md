# 5090 · 方案 24 · 3s/hop100 算法增量（V/T/E 腿）



OpenBMI Three · Tw=3s · hop=100ms · Val Acc_paper 早停 · batch **256/512**。



## 臂



| 脚本 | 用途 |

|------|------|

| `baseline_shallow.py` | 阶段 1 校准（S3）+ V/E 锚 ckpt |

| `baseline_eegnet.py` | E 成员 |

| `baseline_conformer.py` | E 成员 |

| `dump_probs.py` | 重放 ckpt 导出 prob（V/E） |

| `replay_v_weighted_vote.py` | V 臂 val 网格 → test |

| `replay_e_fusion.py` | E 臂三骨干概率融合 |



## 前置（5090 无 3s ckpt 时必做）



**不能**直接用 5070/5060 的 shallow ckpt 做 V/T/E 主判定；须本机重训 S3。



```powershell

conda activate cyy

cd F:\Cyy\MI\code\preprocess_lab

python -m src.datasets.openbmi.batch --config config/openbmi_3s_hop100.yaml

```



## 阶段 1 · S3 校准五折



```powershell

cd F:\Cyy\MI\code\train_lab\src\step\5090_baselines_openbmi_3s_hop100_accpaper

python baseline_shallow.py --three-only --max-folds 0 --num-workers 0

```



校准门：Three ∈ **[0.584, 0.591]**。



## 阶段 2 · V 重放（零训练）



```powershell

$RUN = "F:\Cyy\MI\code\train_lab\out\5090_alg_incr_3s_hop100_accpaper\shallow_...\run_...\three"

python baseline_shallow.py --dump-probs --replay-run-dir $RUN --replay-stage three

python replay_v_weighted_vote.py --run-dir $RUN

```



## 阶段 3 · T 臂 t0 软降权



```powershell

python baseline_shallow.py --three-only --t0-weight 0.6 --max-folds 0 --num-workers 0

# 附报：--t0-weight 0.3 · 硬滤 --t0-filter-max 0.7

```



## 阶段 4 · E 链



```powershell

powershell -File .\run_24_chain_3s_guarded.ps1 -From eegnet -MaxFolds 0 -NoConsole

# 三 run 齐后：

python replay_e_fusion.py --shallow-run $S3 --eegnet-run $EN --conformer-run $CF

```



## out



`code/train_lab/out/5090_alg_incr_3s_hop100_accpaper/`



## W 腿（O3s_m · pf1000）



在 `5090_mech_verify_accpaper`：



```powershell

cd ..\5090_mech_verify_accpaper

powershell -File .\run_24_w_o3s_guarded.ps1 -MaxFolds 0 -NoConsole

python dump_probs_23.py --arm O3s_m --run-dir $O3RUN

python replay_w_adaptive_window.py --o1-run $O1 --o2-run $O2 --o3-run $O3

```

