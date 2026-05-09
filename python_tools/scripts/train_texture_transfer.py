#!/usr/bin/env python3
"""Train the style-conditioned texture transfer U-Net."""
from __future__ import annotations

import argparse
import json
import math
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
from map_processor.models.texture_transfer_unet import TextureTransferUNet  # noqa: E402


class MapCropDataset(Dataset):
    """Random crops with mirror+rotate augmentation. Each item = one crop from one map.

    Per epoch, we sample N_PER_MAP crops per map.
    """

    def __init__(
        self,
        records: List[dict],
        data_root: Path,
        crop: int,
        per_map: int,
        ignore_index: int = -1,
        is_train: bool = True,
    ):
        self.records = records
        self.data_root = data_root
        self.crop = crop
        self.per_map = per_map
        self.ignore_index = ignore_index
        self.is_train = is_train

    def __len__(self):
        return len(self.records) * self.per_map

    def __getitem__(self, idx):
        rec = self.records[idx % len(self.records)]
        d = np.load(self.data_root / rec["npz"])
        X = d["X"]; y = d["y"]; style = int(d["style_id"])
        C, W, H = X.shape

        c = self.crop
        if W >= c and H >= c:
            x0 = random.randint(0, W - c) if self.is_train else max(0, (W - c) // 2)
            y0 = random.randint(0, H - c) if self.is_train else max(0, (H - c) // 2)
            X = X[:, x0:x0 + c, y0:y0 + c]
            y = y[x0:x0 + c, y0:y0 + c]
        else:
            # Pad if smaller than crop
            pad_w = max(0, c - W)
            pad_h = max(0, c - H)
            X = np.pad(X, ((0, 0), (0, pad_w), (0, pad_h)), mode="reflect")
            y = np.pad(y, ((0, pad_w), (0, pad_h)), mode="constant", constant_values=self.ignore_index)
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


def make_class_weights(class_freq: np.ndarray, mode: str = "sqrt", clamp: float = 50.0) -> torch.Tensor:
    """Inverse-frequency class weights."""
    f = class_freq.astype(np.float64) + 1.0
    if mode == "inv":
        w = 1.0 / f
    elif mode == "sqrt":
        w = 1.0 / np.sqrt(f)
    else:
        w = np.ones_like(f)
    w = w / w.mean()
    w = np.clip(w, 0.0, clamp)
    return torch.from_numpy(w).float()


def evaluate(model, loader, device, n_classes: int, ignore_index: int) -> dict:
    model.eval()
    n = 0; correct = 0; correct_top5 = 0
    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device); y = y.to(device); s = s.to(device)
            logits = model(X, s)
            mask = y != ignore_index
            preds = logits.argmax(dim=1)
            correct += int((preds[mask] == y[mask]).sum())
            top5 = logits.topk(5, dim=1).indices  # (B,5,H,W)
            t5 = (top5 == y.unsqueeze(1)).any(dim=1)
            correct_top5 += int(t5[mask].sum())
            n += int(mask.sum())
            yc = y[mask].cpu().numpy(); pc = preds[mask].cpu().numpy()
            for c in np.unique(yc):
                m2 = yc == c
                per_class_total[c] += int(m2.sum())
                per_class_correct[c] += int((pc[m2] == c).sum())
    classes_hit = (per_class_total > 0).sum()
    macro_acc = (per_class_correct[per_class_total > 0] / np.maximum(per_class_total[per_class_total > 0], 1)).mean()
    return {
        "n_eval_pixels": n,
        "top1": correct / max(n, 1),
        "top5": correct_top5 / max(n, 1),
        "macro_acc": float(macro_acc),
        "classes_seen": int(classes_hit),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path,
                    default=_python_tools_root() / "training_outputs" / "texture_transfer")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--per_map", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--weight_mode", choices=["sqrt", "inv", "none"], default="sqrt")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    out_dir = args.out_dir or (args.data_dir / "ckpt")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((args.data_dir / "index.json").read_text(encoding="utf-8"))
    train_recs = [r for r in index["records"] if r["split"] == "train"]
    val_recs = [r for r in index["records"] if r["split"] == "val"]
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}")

    n_chan = index["n_channels"]; n_styles = index["n_styles"]
    vocab_size = index["vocab_size"]; ignore_index = index["ignore_index"]
    class_freq = np.array(index["class_freq_train"], dtype=np.int64)

    train_ds = MapCropDataset(train_recs, args.data_dir, args.crop, args.per_map,
                              ignore_index=ignore_index, is_train=True)
    val_ds = MapCropDataset(val_recs, args.data_dir, args.crop, max(1, args.per_map // 2),
                            ignore_index=ignore_index, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = TextureTransferUNet(in_channels=n_chan, n_styles=n_styles,
                                vocab_size=vocab_size, base=args.base).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    weights = make_class_weights(class_freq, mode=args.weight_mode).to(args.device)
    crit = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    best_top1 = -1.0
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0; nb = 0
        for X, y, s in train_loader:
            X = X.to(args.device); y = y.to(args.device); s = s.to(args.device)
            logits = model(X, s)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += float(loss.item()); nb += 1
        train_loss = running / max(nb, 1)
        metrics = evaluate(model, val_loader, args.device, vocab_size, ignore_index)
        line = {
            "epoch": epoch, "train_loss": train_loss,
            "val_top1": metrics["top1"], "val_top5": metrics["top5"],
            "val_macro_acc": metrics["macro_acc"], "classes_seen": metrics["classes_seen"],
            "lr": opt.param_groups[0]["lr"],
        }
        log_f.write(json.dumps(line) + "\n"); log_f.flush()
        print(f"epoch {epoch:3d}  loss={train_loss:.4f}  top1={metrics['top1']:.4f}  "
              f"top5={metrics['top5']:.4f}  macro={metrics['macro_acc']:.4f}  "
              f"classes={metrics['classes_seen']}/{vocab_size}")

        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "metrics": metrics,
                "epoch": epoch,
            }, out_dir / "best.pt")
            print(f"  saved best.pt (top1={best_top1:.4f})")

    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs},
               out_dir / "last.pt")
    log_f.close()
    print(f"Done. Best val top1: {best_top1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
