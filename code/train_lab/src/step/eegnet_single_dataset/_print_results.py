import json
from pathlib import Path

root = Path(r"D:\cyy\MI\code\train_lab\out\eegnet_single_dataset")
paths = sorted(root.rglob("task/summary.json"))
if not paths:
    print("no summaries yet")
for s in paths:
    d = json.loads(s.read_text(encoding="utf-8"))
    print("===", d["model_name"], d["data_tag"], "===")
    print("N", d.get("n_trials"), d.get("class_counts"), "n_times", d.get("n_times"))
    for k in [
        "test_acc",
        "test_specificity",
        "test_recall",
        "test_precision",
        "test_f1",
        "test_balanced_accuracy",
        "val_f1",
    ]:
        print(f"  {k}: {d[k + '_mean']:.4f} +/- {d[k + '_std']:.4f}")
    print("  folds:")
    for f in d["folds"]:
        m = f["test_metrics"]
        print(
            f"    fold{f['fold']}: Acc={m['accuracy']:.3f} Spec={m['specificity']:.3f} "
            f"Rec={m['recall']:.3f} Prec={m['precision']:.3f} F1={m['f1']:.3f} "
            f"BalAcc={m['balanced_accuracy']:.3f}"
        )
    print()
