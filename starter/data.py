# The training data the baseline learned from: 15M Lichess games as token ids, the same
# vocabulary as vocabulary.json. Downloads once from Hugging Face (2.3 GB), then serves
# random blocks the way the baseline was trained (blocks of 256 tokens).
#   from data import batches
#   for x, y in batches("train", batch_size=64):   # x, y: int64 [64, 256]; y is x shifted by one
import numpy as np
import torch
from huggingface_hub import hf_hub_download

REPO = "Sparshith/nanodanya-games-15m"


def tokens(split="train"):
    # uint16 ids. Games follow each other: <bos> moves <eos> <bos> moves <eos> ...
    path = hf_hub_download(REPO, f"{split}.bin", repo_type="dataset")
    return np.memmap(path, dtype=np.uint16, mode="r")


def batches(split="train", batch_size=64, block=256, seed=0):
    data = tokens(split)
    rng = np.random.default_rng(seed)
    while True:
        starts = rng.integers(0, len(data) - block - 1, size=batch_size)
        x = np.stack([data[s:s + block] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1:s + block + 1] for s in starts]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)
