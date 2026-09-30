# Starter

The baseline, ready to run, train, or replace. Nine layers, width 320, 15.8 million weights, 9.45 MB. It scores 1303 ±16 internal Elo and solves 22.1% of puzzles without legal-move masking. Trained from the data in this folder with the script in this folder. Your job is to make it better, or to make something better.

| File | What it is |
|---|---|
| `model.py` | The baseline in PyTorch, about 100 lines. In: token ids `[batch, tokens]`. Out: logits `[batch, tokens, vocabulary]`. Train it, change it, or replace it with a causal next-token decoder that has the same input and output. |
| `baseline.pt` | The baseline's weights, 9 MB. `export.py` downloads it the first time. Gitignored. |
| `train.py` | The recipe that made the baseline: 200k steps on `data.py`, then 2k steps with 4-bit rounding. Run it as is to reproduce the baseline, or change anything. |
| `export.py` | Turns weights into `model.onnx`, the one file you drop on the page. Writes the vocabulary and context length into the file, then prints the size against the 10 MB limit. |
| `vocabulary.json` | The token list: `<bos>`, `<eos>`, then one chess move per token in standard notation (`e4`, `Nf3`, `O-O`, `Qxf7#`). A token's id is its position in the list. The baseline was trained with it. You may edit it or bring your own; it must contain `<bos>`. |
| `check.py` | The page's three checks, on your laptop, plus a quick match against the baseline. Run it after every export. Go to the page only for the file you want to submit. |
| `data.py` | The training data the baseline learned from: 15M Lichess games as token ids on Hugging Face. Downloads once (2.3 GB) and serves random 256-token blocks, the way the baseline was trained. |
| `tokenize_game.py` | Example: turns a game's moves into token ids with this vocabulary. Use the same rule when you bring your own games. |
| `requirements.txt` | torch, onnx, onnxruntime, python-chess, huggingface_hub, pinned. |

## Minute one

```sh
pip install -r requirements.txt
python3 export.py --int4
python3 check.py
```

`export.py` writes `model.onnx`, 9.45 MB. `check.py` runs the page's three checks on it and plays 10 games against the baseline. Both sides use the same weights, but illegal predictions lose immediately, so games need not be draws. Drop the file on the challenge page when you want to see it play on a board.

## Train

`python3 train.py` reproduces the baseline: AdamW, batch 64 of 256 tokens, cosine 3e-4 to 3e-5 over 200k steps, then 2k steps at a tenth of that with 4-bit rounding on. About a day on one GPU. `--steps 200 --qat-steps 20` is a smoke test.

Or write your own loop. Load the baseline and keep training, or start fresh:

```python
from model import Model, load_weights

config, state, _ = load_weights("baseline.pt")
model = Model(config)            # qat=True by default: rounds weights to 4 bits during training
model.load_state_dict(state)     # skip this line to start from scratch
# ... your data, your loop ...
torch.save(model.state_dict(), "mine.pt")
```

The baseline's own training data is one import away:

```python
from data import batches

for x, y in batches("train", batch_size=64):   # int64 [64, 256]; y is x shifted by one token
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    ...
```

That is 15M Lichess games from 2025, 1.05 billion tokens, in this vocabulary, from https://huggingface.co/datasets/Sparshith/nanodanya-games-15m. The model predicts the next id from the ids so far. That is the whole objective. Want other games? Lichess publishes every month at https://database.lichess.org; turn them into ids the way `tokenize_game.py` does.

`Model(config)` has `qat=True`: during training the forward pass rounds every weight to 4 bits, so what you train is what `--int4` exports. Pass `qat=False` if you plan to export at fp32 or int8.

## Export and test

```sh
python3 export.py --checkpoint mine.pt --name "Mine v3" --int4
python3 check.py
```

Precision is your lever against the 10 MB limit:

| Flag | Bits per weight | Fits about |
|---|---|---|
| none | 32 | 2.5M weights |
| `--int8` | 8 | 10M weights |
| `--int4` | 4 | 20M weights |

`--int4` packs every 2-D weight matrix whose width is a multiple of 64 and adds a few standard ONNX ops that unpack it at load time. It works for the baseline architecture and for most others exported through PyTorch. If it prints TOO BIG, use a smaller precision or a smaller model.

`check.py` prints the size, the load, the four test positions, and the match score. Iterate here. When a file wins, drop it on the page, run the 20-game match there too, and click Submit.

## How the page talks to your model

1. It reads the vocabulary from the file's metadata.
2. It loads the file with ONNX Runtime Web, in the visitor's browser.
3. For every move, it turns the game so far into your token ids and runs the model once.
4. It selects the highest-scoring token across your entire vocabulary (lowest token id breaks ties). If that token is not a legal SAN move, you lose immediately. No masking, retries, or fallback moves. Check and mate suffixes are optional; the rest of the SAN must match.

Your model must learn both legal moves and good moves. Predicting a special token such as `<eos>` on a live turn also loses the game.

## Limits

- One ONNX file, at most 10,000,000 bytes. Standard ops only, opset 17 or newer.
- Context length from 64 to 4,096 tokens.
- No search, no chess engine, no hand-written chess rules at inference time.
- Training: anything goes.

Full contract: `../llms.txt`.
