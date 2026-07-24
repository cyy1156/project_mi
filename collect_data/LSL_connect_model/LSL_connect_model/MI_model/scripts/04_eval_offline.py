"""步骤 4：离线评估（混淆矩阵 + 逐类指标）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MI_ROOT = Path(__file__).resolve().parent.parent
if str(MI_ROOT) not in sys.path:
    sys.path.insert(0, str(MI_ROOT))

from config import STAGES, dataset_dir, models_dir, reports_dir


def evaluate_stage(stage_id: int) -> Path:
    stage = STAGES[stage_id]
    ds_dir = dataset_dir(stage)
    m_dir = models_dir(stage)
    r_dir = reports_dir(stage)
    r_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(ds_dir / "X.npy")
    y = np.load(ds_dir / "y.npy")
    test_idx = np.load(ds_dir / "test_idx.npy")
    meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
    class_names = meta["class_names"]
    split_info = meta.get("train_test_split", {})

    X_test, y_test = X[test_idx], y[test_idx]

    pipe_path = m_dir / "pipeline.pkl"
    if pipe_path.is_file():
        pipe = joblib.load(pipe_path)
        y_pred = pipe.predict(X_test)
    else:
        csp = joblib.load(m_dir / "csp.pkl")
        svm = joblib.load(m_dir / "svm.pkl")
        feats = csp.transform(X_test)
        y_pred = svm.predict(feats)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )

    cv_path = r_dir / "cv_results.json"
    cv_summary = None
    if cv_path.is_file():
        cv_summary = json.loads(cv_path.read_text(encoding="utf-8"))

    metrics = {
        "stage": stage.name,
        "test_set_accuracy": acc,
        "n_test_samples": int(len(test_idx)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_names": class_names,
        "train_test_split": split_info,
        "cv_summary": cv_summary,
    }
    metrics_path = r_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title=f"{stage.name} confusion matrix (held-out test acc={acc:.3f})",
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    cm_path = r_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    print(f"测试集准确率 (20% hold-out): {acc:.4f}  (n={len(test_idx)})")
    print(classification_report(y_test, y_pred, target_names=class_names))
    if cv_summary:
        print(
            f"训练集内 CV: {cv_summary['mean_accuracy']:.4f} "
            f"± {cv_summary['std_accuracy']:.4f}"
        )
    print(f"指标: {metrics_path}")
    print(f"混淆矩阵图: {cm_path}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="离线评估")
    parser.add_argument("--stage", type=int, default=1, choices=sorted(STAGES.keys()))
    args = parser.parse_args()
    evaluate_stage(args.stage)


if __name__ == "__main__":
    main()
