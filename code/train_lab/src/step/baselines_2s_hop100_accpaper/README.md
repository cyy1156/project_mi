# baselines_2s_hop100_accpaper

> **现行 03 重训包**：Val / Test 主指标 = **Acc_paper**；train **batch balance**；仅 **BCI2a T**；**无 RAP**。  
> 与下列包 **隔离**，勿混用权重或口径：  
> - `baselines_2s_hop100/` — 窗级 Val **BalAcc** + balbatch  
> - `baselines_2s_hop100_trialmaj/` — 历史 no_retrain 复评  

方案：[`资料/模型训练/03_旁路_2s滑窗100ms_试次多数票/方案.md`](../../../../../资料/模型训练/03_旁路_2s滑窗100ms_试次多数票/方案.md)

## 用法（与 01 同形）

```bash
cd code/train_lab/src/step/baselines_2s_hop100_accpaper

python trial_metrics.py
python baseline_eegnet.py
python baseline_dbn.py --skip-three
python run_all.py --continue-on-error
python run_all.py --smoke --continue-on-error
# 兼容旧入口
python train_one.py --model eegnet
```

模型脚本：`baseline_{eegnet,shallow,deep,eegtcnet,conformer,dbn,gcbnet,dgcnn,dbn_raw,gcbnet_raw,dgcnn_raw}.py`  
网络结构与 01 对齐；`feat_bandpower` / `load_external` / `raw_time` 复用 01（经 `_hop100_path.py` append）。

## 输出

- 权重：`train_lab/out/baseline_2s_hop100_accpaper/<model>_2s_hop100_balbatch_accpaper/bci2a_2s_hop100/run_<stamp>/`
- MD：`资料/模型训练/runs/<stamp>_<model>_2s_hop100_balbatch_accpaper/`

与 01 的差异：**早停/选模 = Acc_paper**（不是窗级 BalAcc）；balbatch 相同。
