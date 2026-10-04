"""Train the trajectory model.

    python scripts/train.py --data "data/raw/*.json" --out checkpoints/model.pt

Fine-tune a model pretrained on a public dataset on your own recordings (architecture and
delta scaling come from the checkpoint, a .pt or the bundled humanmouse/assets/model.npz; use a lower
learning rate):

    python scripts/train.py --init checkpoints/pretrained.pt --data "data/raw/*.json" --lr 2e-4 --epochs 30
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanmouse.data import BucketBatchSampler, TrajectoryDataset, collate, load_trials  # noqa: E402
from humanmouse.generate import load_checkpoint  # noqa: E402
from humanmouse.model import MouseMDN  # noqa: E402


def run_epoch(model, loader, device, opt=None):
    model.train(opt is not None)
    totals, steps = np.zeros(3), 0.0
    with torch.set_grad_enabled(opt is not None):
        for x, d, end, mask in loader:
            x, d, end, mask = x.to(device), d.to(device), end.to(device), mask.to(device)
            params, _ = model(x)
            loss, nll, bce = model.loss(params, d, end, mask)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            n = mask.sum().item()
            totals += np.array([loss.item(), nll.item(), bce.item()]) * n
            steps += n
    return totals / steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/*.json")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mixtures", type=int, default=10)
    p.add_argument("--init", help="checkpoint to fine-tune from (overrides --hidden/--layers/--mixtures)")
    p.add_argument("--jitter-px", type=float, default=0.5, help="dequantization noise; 0 to disable")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    init = None
    if args.init:
        weights, config, stats = load_checkpoint(args.init)
        init = {"model": {k: torch.from_numpy(v) for k, v in weights.items()}, "config": config, "stats": stats}

    trials = load_trials(args.data)
    if not trials:
        sys.exit(f"no trials found in {args.data}")
    random.shuffle(trials)
    n_val = max(1, int(len(trials) * args.val_frac))
    train_ds = TrajectoryDataset(trials[n_val:], stats=init["stats"] if init else None, jitter_px=args.jitter_px)
    val_ds = TrajectoryDataset(trials[:n_val], stats=train_ds.stats, jitter_px=args.jitter_px)
    print(f"{len(trials)} trials -> {len(train_ds)} train / {len(val_ds)} val usable")

    train_dl = DataLoader(train_ds, batch_sampler=BucketBatchSampler(train_ds, args.batch, seed=args.seed), collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_sampler=BucketBatchSampler(val_ds, args.batch, shuffle=False), collate_fn=collate)

    if init:
        model = MouseMDN(**init["config"])
        model.load_state_dict(init["model"])
        print(f"fine-tuning from {args.init} ({init['config']})")
    else:
        model = MouseMDN(hidden=args.hidden, layers=args.layers, mixtures=args.mixtures)
    model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def validate():
        val_ds.rng = np.random.default_rng(args.seed)  # same noise every time so epochs are comparable
        return run_epoch(model, val_dl, args.device)

    best = float("inf")
    if init:  # the starting point is a valid result; later epochs must beat it
        best = validate()[0]
        torch.save({"model": model.state_dict(), "config": model.config, "stats": train_ds.stats}, args.out)
        print(f"initial val loss {best:.4f}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_dl, args.device, opt)
        va = validate()
        sched.step()
        mark = ""
        if va[0] < best:
            best = va[0]
            torch.save({"model": model.state_dict(), "config": model.config, "stats": train_ds.stats}, args.out)
            mark = "  *saved"
        print(f"epoch {epoch:3d}  train {tr[0]:.4f} (nll {tr[1]:.4f} end {tr[2]:.4f})  val {va[0]:.4f} (nll {va[1]:.4f} end {va[2]:.4f})  {time.perf_counter() - t0:.0f}s{mark}", flush=True)
    print(f"best val loss {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
