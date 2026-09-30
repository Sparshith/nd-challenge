# Baseline

The model to beat: 1303 ±16 internal Elo, 1941/8800 puzzles solved (22.1%), 9.45 MB as an ONNX file, with illegal predictions forfeiting immediately. Its previous masked result was 1517 ±12 Elo and 23.0% puzzles; those games are excluded from the new rating pool. A 15.8M-parameter transformer (9 layers, width 320, tied embeddings) trained for 200k steps on the public 15M-game dataset with no teacher, then 2k steps of 4-bit-aware tuning, with every matrix stored in 4 bits.

The baseline is the starter. Its architecture is `../starter/model.py`, its recipe is `../starter/train.py`, its data is `../starter/data.py`, its weights are `../starter/baseline.pt`, and `../starter/export.py --int4` produces the exact file the page serves. The previous baseline (6 layers, width 384, distilled, 1273 Elo) is the `baseline-v1` release.

This folder holds one thing:

| File | What it is |
|---|---|
| `test_export.py` | Exports the baseline with the starter script and checks its logits against the PyTorch model and against the served file on 1, 8 and 64-token inputs. |

Run it with `python3 -m pytest baseline/test_export.py`.

Unmasked evaluation: 1,200 games, 200 openings with both colors, greedy predictions, 691 illegal/non-move forfeits. The current browser file has SHA-256 `3abfac3fd6d8f859918ec9f0366399d582b9996a1564d17e3ad1c014a91562b6`. See [evaluation](evaluation.md).
