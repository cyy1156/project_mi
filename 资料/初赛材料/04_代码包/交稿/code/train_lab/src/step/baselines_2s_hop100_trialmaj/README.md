# baselines_2s_hop100_trialmaj（历史 · 只读）

> **【历史包 · no_retrain】**  
> 复用 `baselines_2s_hop100` 已训权重做试次级 **复评**。  
> **不是**现行 03 的 Acc_paper 早停重训。  
> 现行重训请用：[`../baselines_2s_hop100_accpaper/`](../baselines_2s_hop100_accpaper/)  
> 方案：[`资料/模型训练/03_旁路_2s滑窗100ms_试次多数票/方案.md`](../../../../../资料/模型训练/03_旁路_2s滑窗100ms_试次多数票/方案.md)

**禁止**：在本目录加 Acc_paper 训练逻辑；勿与 `*_accpaper` 混用权重/口径。

```bash
cd code/train_lab/src/step/baselines_2s_hop100_trialmaj
python trial_aggregate.py
python reeval_kfold.py --model eegnet --smoke-fold 0
python run_all_reeval.py --continue-on-error
```

输出：`train_lab/out/baseline_2s_hop100_trialmaj/...`
