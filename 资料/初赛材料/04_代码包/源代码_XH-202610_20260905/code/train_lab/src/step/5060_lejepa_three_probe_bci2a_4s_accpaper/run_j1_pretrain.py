"""J1 · LeJEPA 预训（BCI2a 4s · L_pred + λ·SIGReg；无 EMA；按折排除 test 被试）。

用法:
  cd code/train_lab/src/step/5060_lejepa_three_probe_bci2a_4s_accpaper
  python run_j1_pretrain.py --max-folds 1 --pretrain-epochs 5 --batch 128
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch

from common import (
    build_backbone,
    load_bci2a,
    make_loader,
    materialize_windows,
    out_root,
    save_json,
    seed_everything,
    subject_masks,
)
from shared_hparams import SHARED
from src.common.steps.split_subjects import iter_subject_kfold


def main():
    p = argparse.ArgumentParser(description="J1 LeJEPA pretrain BCI2a 4s (fold-safe)")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--pretrain-epochs", type=int, default=0)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--lambda-sigreg", type=float, default=-1.0, help=">=0 覆盖 λ")
    p.add_argument("--resume-dir", type=str, default="", help="已有 run_ 目录则写入其 j1/")
    p.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help=">0 时每折最多用 N 个窗做冒烟（随机子采样；BCI2a 全库仅 ~1719，一般不用）",
    )
    args = p.parse_args()

    hp = SHARED
    repl = {}
    if args.pretrain_epochs > 0:
        repl["pretrain_epochs"] = args.pretrain_epochs
    if args.batch > 0:
        repl["pretrain_batch"] = args.batch
    if args.num_workers >= 0:
        repl["num_workers"] = args.num_workers
    if args.lambda_sigreg >= 0:
        repl["lambda_sigreg"] = float(args.lambda_sigreg)
    if repl:
        hp = replace(hp, **repl)

    seed_everything(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[j1] loading data device={device} …", flush=True)
    data = load_bci2a(hp)
    X, subjects = data["X"], data["subjects"]
    print(
        f"[j1] X{tuple(X.shape)} loaded  trial_id_synthetic={data['trial_id_synthetic']}",
        flush=True,
    )

    if args.resume_dir:
        from pathlib import Path

        root = Path(args.resume_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = out_root()
    j1_dir = root / "j1"
    j1_dir.mkdir(parents=True, exist_ok=True)
    print(f"[j1] out={root} device={device} epochs={hp.pretrain_epochs}", flush=True)

    fold_summaries = []
    for i, info in enumerate(
        iter_subject_kfold(
            subjects, n_folds=hp.n_folds, val_ratio=hp.val_ratio, seed=hp.seed
        )
    ):
        if args.max_folds > 0 and i >= args.max_folds:
            break
        fold = int(info["fold"])
        masks = subject_masks(
            subjects, info["train_subjects"], info["val_subjects"], info["test_subjects"]
        )
        tr_idx = np.flatnonzero(masks["train"] | masks["val"])
        if args.max_windows > 0 and len(tr_idx) > args.max_windows:
            rng = np.random.default_rng(hp.seed + fold)
            tr_idx = np.sort(rng.choice(tr_idx, size=int(args.max_windows), replace=False))
            print(f"[j1] smoke subsample → {len(tr_idx)} windows", flush=True)
        print(
            f"\n======== J1 fold{fold} n_win={len(tr_idx)} "
            f"test_subj={info['test_subjects']} ========",
            flush=True,
        )
        print(f"[j1] materialize {len(tr_idx)} windows → RAM …", flush=True)
        ds = materialize_windows(X, tr_idx, y=None)
        loader = make_loader(ds, hp.pretrain_batch, shuffle=True, hp=hp)

        seed_everything(hp.seed + fold)
        model = build_backbone(hp).to(device)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=hp.pretrain_lr,
            weight_decay=hp.weight_decay,
        )
        n_tok = hp.n_chans * (hp.n_times_expected // hp.patch_time)
        print(
            f"[j1] model on {device}, n_tokens={n_tok}, "
            f"steps/epoch≈{max(1, len(ds)//hp.pretrain_batch)}",
            flush=True,
        )

        for ep in range(1, hp.pretrain_epochs + 1):
            model.train()
            total, n = 0.0, 0
            last_info = {}
            for bi, xb in enumerate(loader, 1):
                xb = xb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if hp.use_amp and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        loss, last_info = model.forward_lejepa(xb)
                    loss.backward()
                else:
                    loss, last_info = model.forward_lejepa(xb)
                    loss.backward()
                opt.step()
                total += float(loss.detach().item()) * xb.size(0)
                n += xb.size(0)
                if bi == 1 or bi % 20 == 0:
                    print(
                        f"  ep{ep} step {bi}/{max(1,len(loader))}  "
                        f"loss={float(loss.detach()):.4f}  "
                        f"pred={last_info.get('l_pred',0):.3f}  "
                        f"sig={last_info.get('l_sigreg',0):.3f}",
                        flush=True,
                    )
            print(
                f"fold{fold} ep {ep:03d}  L={total/max(n,1):.4f}  "
                f"pred={last_info.get('l_pred',0):.3f}  "
                f"sigreg={last_info.get('l_sigreg',0):.3f}  "
                f"z_std={last_info.get('z_std',0):.3f}",
                flush=True,
            )

        ckpt = {
            "fold": fold,
            "arm": "j1_lejepa_pretrain",
            "hparams": {**{k: getattr(hp, k) for k in hp.__dataclass_fields__}},
            "test_subjects": list(map(str, info["test_subjects"])),
            "train_subjects": list(map(str, info["train_subjects"])),
            "val_subjects": list(map(str, info["val_subjects"])),
            "backbone": model.state_dict(),
        }
        path = j1_dir / f"fold{fold}_lejepa.pt"
        torch.save(ckpt, path)
        fold_summaries.append(
            {"fold": fold, "ckpt": str(path), "n_pretrain_windows": int(len(tr_idx))}
        )
        print(f"  saved {path}")

    save_json(
        j1_dir / "summary.json",
        {
            "arm": "j1",
            "data_tag": hp.data_tag,
            "stamp": root.name,
            "device": str(device),
            "hparams": {k: getattr(hp, k) for k in hp.__dataclass_fields__},
            "folds": fold_summaries,
            "finished": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"\n[j1] done → {j1_dir}")


if __name__ == "__main__":
    main()
