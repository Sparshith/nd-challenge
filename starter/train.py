# The recipe that made the baseline. Run as is to reproduce it (one GPU, about a day), or
# change anything. Saves best.pt by validation loss; export with `export.py --checkpoint best.pt --int4`.
#   python3 train.py                         full run: 200k float steps, then 2k steps of 4-bit-aware tuning
#   python3 train.py --steps 200 --qat-steps 20   smoke test
import argparse
import math
import time

import torch
import torch.nn.functional as F

from data import batches
from model import Config, Model

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200_000)
parser.add_argument("--qat-steps", type=int, default=2_000, help="extra steps with 4-bit rounding in the forward pass")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--block", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval-every", type=int, default=1_000)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = Model(Config(), qat=False).to(device)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}")
train = batches("train", args.batch, args.block, seed=args.seed)
val = [next(batches("val", args.batch, args.block, seed=314159)) for _ in range(4)]


def loss_on(x, y):
    return F.cross_entropy(model(x.to(device)).flatten(0, 1), y.to(device).flatten())


def run(steps, lr, min_lr, warmup):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, fused=device == "cuda")
    best, start = float("inf"), time.time()
    for step in range(1, steps + 1):
        progress = (step - warmup) / max(1, steps - warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr * step / warmup if step < warmup else min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
        model.train()
        loss = loss_on(*next(train))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_loss = sum(loss_on(x, y).item() for x, y in val) / len(val)
            print(f"step {step:>7}  train {loss.item():.3f}  val {val_loss:.3f}  {time.time() - start:.0f}s")
            if val_loss < best:
                best = val_loss
                torch.save(model.state_dict(), "best.pt")
    torch.save(model.state_dict(), "last.pt")


run(args.steps, args.lr, args.lr / 10, warmup=200)
if args.qat_steps:
    model.qat = True
    model.load_state_dict(torch.load("best.pt", map_location=device))
    run(args.qat_steps, args.lr / 10, args.lr / 100, warmup=0)
print("done: best.pt is the checkpoint to export")
