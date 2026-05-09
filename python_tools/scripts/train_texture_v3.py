#!/usr/bin/env python3
"""Train v3: SPADE U-Net + Focal Frequency Loss + LPIPS perceptual loss.

Reads curated_index.json (output of curate_training_set.py).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
from map_processor.models.spade_texture_unet import SPADEUNet  # noqa: E402
from map_processor.models.texture_losses import TextureAuxLosses  # noqa: E402


class MapCropDataset(Dataset):
    def __init__(self, records, data_root, crop, per_map, ignore_index=-1, is_train=True):
        self.records = records; self.data_root = data_root; self.crop = crop
        self.per_map = per_map; self.ignore_index = ignore_index; self.is_train = is_train

    def __len__(self):
        return len(self.records) * self.per_map

    def __getitem__(self, idx):
        rec = self.records[idx % len(self.records)]
        d = np.load(self.data_root / rec["npz"])
        X = d["X"]; y = d["y"]; style = int(d["style_id"])
        C, W, H = X.shape; c = self.crop
        if W >= c and H >= c:
            x0 = random.randint(0, W - c) if self.is_train else max(0, (W - c) // 2)
            y0 = random.randint(0, H - c) if self.is_train else max(0, (H - c) // 2)
            X = X[:, x0:x0 + c, y0:y0 + c]; y = y[x0:x0 + c, y0:y0 + c]
        else:
            pw = max(0, c - W); ph = max(0, c - H)
            X = np.pad(X, ((0, 0), (0, pw), (0, ph)), mode="reflect")
            y = np.pad(y, ((0, pw), (0, ph)), mode="constant", constant_values=self.ignore_index)
            X = X[:, :c, :c]; y = y[:c, :c]
        if self.is_train:
            if random.random() < 0.5:
                X = np.flip(X, axis=1).copy(); y = np.flip(y, axis=0).copy()
            if random.random() < 0.5:
                X = np.flip(X, axis=2).copy(); y = np.flip(y, axis=1).copy()
            k = random.randint(0, 3)
            if k:
                X = np.rot90(X, k=k, axes=(1, 2)).copy()
                y = np.rot90(y, k=k, axes=(0, 1)).copy()
        return torch.from_numpy(X).float(), torch.from_numpy(y).long(), torch.tensor(style, dtype=torch.long)


def make_class_weights(class_freq, mode="sqrt", clamp=50.0):
    f = class_freq.astype(np.float64) + 1.0
    if mode == "inv":
        w = 1.0 / f
    elif mode == "sqrt":
        w = 1.0 / np.sqrt(f)
    else:
        w = np.ones_like(f)
    w = w / w.mean()
    return torch.from_numpy(np.clip(w, 0.0, clamp)).float()


def evaluate(model, loader, device, n_classes, ignore_index):
    model.eval()
    n = 0; correct = 0; correct_top5 = 0
    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device); y = y.to(device); s = s.to(device)
            logits = model(X, s)
            mask = y != ignore_index
            preds = logits.argmax(dim=1)
            correct += int((preds[mask] == y[mask]).sum())
            top5 = logits.topk(5, dim=1).indices
            t5 = (top5 == y.unsqueeze(1)).any(dim=1)
            correct_top5 += int(t5[mask].sum())
            n += int(mask.sum())
    return {"top1": correct / max(n, 1), "top5": correct_top5 / max(n, 1), "n_eval_pixels": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path,
                    default=_python_tools_root() / "training_outputs" / "texture_transfer")
    ap.add_argument("--index_file", type=str, default="curated_index.json")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--per_map", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--ffl_weight", type=float, default=0.2)
    ap.add_argument("--lpips_weight", type=float, default=0.1)
    ap.add_argument("--ce_weight", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--weight_mode", choices=["sqrt", "inv", "none"], default="sqrt")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    out_dir = args.out_dir or (args.data_dir / "ckpt_v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.data_dir / args.index_file
    if not index_path.exists():
        raise SystemExit(f"Missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    train_recs = [r for r in index["records"] if r["split"] == "train"]
    val_recs = [r for r in index["records"] if r["split"] == "val"]
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}  ({args.index_file})")

    n_chan = index["n_channels"]; n_styles = index["n_styles"]
    vocab_size = index["vocab_size"]; ignore_index = index["ignore_index"]
    class_freq = np.array(index["class_freq_train"], dtype=np.int64)
    vocab = json.loads((args.data_dir / "vocab.json").read_text(encoding="utf-8"))

    train_ds = MapCropDataset(train_recs, args.data_dir, args.crop, args.per_map, ignore_index, True)
    val_ds = MapCropDataset(val_recs, args.data_dir, args.crop, max(1, args.per_map // 2), ignore_index, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = SPADEUNet(in_channels=n_chan, n_styles=n_styles,
                      vocab_size=vocab_size, base=args.base).to(args.device)
    aux = TextureAuxLosses(vocab_size=vocab_size, vocab=vocab,
                           ffl_weight=args.ffl_weight, lpips_weight=args.lpips_weight,
                           ignore_index=ignore_index).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_aux = sum(p.numel() for p in aux.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.2f}M, aux trainable: {n_aux/1e3:.1f}K")

    weights = make_class_weights(class_freq, mode=args.weight_mode).to(args.device)
    crit = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    # Optimizer covers main model + the small color embedding inside aux
    params = list(model.parameters()) + [p for p in aux.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(len(train_loader), 1))

    best_top1 = -1.0
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(1, args.epochs + 1):
            model.train(); aux.train()
            running = {"ce": 0.0, "aux": 0.0, "n": 0}
            for X, y, s in train_loader:
                X = X.to(args.device); y = y.to(args.device); s = s.to(args.device)
                logits = model(X, s)
                ce = crit(logits, y)
                ax = aux(logits, y)
                loss = args.ce_weight * ce + ax
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                running["ce"] += float(ce.item())
                running["aux"] += float(ax.item())
                running["n"] += 1
            n = max(running["n"], 1)
            tl_ce = running["ce"] / n; tl_aux = running["aux"] / n
            metrics = evaluate(model, val_loader, args.device, vocab_size, ignore_index)
            line = {"epoch": epoch, "ce": tl_ce, "aux": tl_aux,
                    "val_top1": metrics["top1"], "val_top5": metrics["top5"],
                    "lr": opt.param_groups[0]["lr"]}
            log_f.write(json.dumps(line) + "\n"); log_f.flush()
            print(f"epoch {epoch:3d}  ce={tl_ce:.4f}  aux={tl_aux:.4f}  "
                  f"top1={metrics['top1']:.4f}  top5={metrics['top5']:.4f}")
            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                torch.save({
                    "model": model.state_dict(),
                    "aux": aux.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                    "epoch": epoch,
                    "vocab_size": vocab_size,
                    "n_styles": n_styles,
                    "n_channels": n_chan,
                    "base": args.base,
                }, out_dir / "best.pt")
                print(f"  saved best.pt (top1={best_top1:.4f})")

    print(f"Done. Best val top1: {best_top1:.4f}  ->  {out_dir/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
