"""步骤 3：CSP + SVM 训练与 5 折 GroupKFold。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

MI_ROOT = Path(__file__).resolve().parent.parent
if str(MI_ROOT) not in sys.path:
    sys.path.insert(0, str(MI_ROOT))

from config import CSP_N_COMPONENTS, CV_N_SPLITS, STAGES, SVM_C, dataset_dir, models_dir, reports_dir


def _make_pipeline(n_classes: int) -> Pipeline:
    n_comp = min(CSP_N_COMPONENTS, n_classes * 2)
    if n_comp % 2 != 0:
        n_comp -= 1
    n_comp = max(n_comp, 2)

    return Pipeline(
        [
            (
                "csp",
                CSP(n_components=n_comp, reg=None, log=True, norm_trace=False),
            ),
            ("svm", SVC(kernel="rbf", C=SVM_C)),
        ]
    )


def train_stage(stage_id: int) -> Path:
    stage = STAGES[stage_id]
    ds_dir = dataset_dir(stage)
    m_dir = models_dir(stage)
    r_dir = reports_dir(stage)
    m_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(ds_dir / "X.npy")
    y = np.load(ds_dir / "y.npy")
    groups = np.load(ds_dir / "groups.npy")
    train_idx = np.load(ds_dir / "train_idx.npy")
    test_idx = np.load(ds_dir / "test_idx.npy")
    meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))

    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    n_classes = len(meta["class_names"])
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        raise RuntimeError(
            f"训练集标签仅含 {unique_labels.tolist()}，CSP 至少需要 2 类。"
            "请先运行 02_build_dataset 并确保每类有可用 CSV。"
        )
    n_train_groups = len(np.unique(g_train))
    n_splits = min(CV_N_SPLITS, n_train_groups)
    if n_splits < 2:
        raise RuntimeError(f"训练集分组数 {n_train_groups} 不足，无法交叉验证")

    split_info = meta.get("train_test_split", {})
    print(
        f"训练集: {len(train_idx)} 样本 / 测试集: {len(test_idx)} 样本 "
        f"（测试集 {split_info.get('test_size', 0.2):.0%} 仅用于步骤 4）"
    )

    gkf = GroupKFold(n_splits=n_splits)
    fold_scores: list[float] = []

    for fold, (cv_train, cv_val) in enumerate(gkf.split(X_train, y_train, g_train)):
        pipe = _make_pipeline(n_classes)
        pipe.fit(X_train[cv_train], y_train[cv_train])
        acc = float(pipe.score(X_train[cv_val], y_train[cv_val]))
        fold_scores.append(acc)
        print(f"  Fold {fold + 1}/{n_splits}: accuracy = {acc:.4f} (train 内 CV)")

    mean_acc = float(np.mean(fold_scores))
    std_acc = float(np.std(fold_scores))
    print(f"\nCV (仅训练集): {mean_acc:.4f} ± {std_acc:.4f} ({n_splits}-fold GroupKFold)")

    final_pipe = _make_pipeline(n_classes)
    final_pipe.fit(X_train, y_train)

    csp = final_pipe.named_steps["csp"]
    svm = final_pipe.named_steps["svm"]
    joblib.dump(csp, m_dir / "csp.pkl")
    joblib.dump(svm, m_dir / "svm.pkl")
    joblib.dump(final_pipe, m_dir / "pipeline.pkl")

    cv_results = {
        "stage": stage.name,
        "n_samples_total": int(X.shape[0]),
        "n_train_samples": int(len(train_idx)),
        "n_test_samples": int(len(test_idx)),
        "n_train_groups": n_train_groups,
        "n_splits": n_splits,
        "fold_accuracies": fold_scores,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "cv_scope": "train_set_only",
        "csp_n_components": csp.n_components,
        "svm_C": SVM_C,
        "class_names": meta["class_names"],
        "train_test_split": split_info,
    }
    cv_path = r_dir / "cv_results.json"
    cv_path.write_text(json.dumps(cv_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"模型已保存: {m_dir}")
    print(f"CV 结果: {cv_path}")
    return m_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="CSP+SVM 训练")
    parser.add_argument("--stage", type=int, default=1, choices=sorted(STAGES.keys()))
    args = parser.parse_args()
    train_stage(args.stage)


if __name__ == "__main__":
    main()
