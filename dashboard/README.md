# Challenge dashboard

The page that checks and plays an entry in the browser. Entrants do not need this folder; they work in `starter/`.

## Run the page locally

```sh
cd dashboard
npm ci --ignore-scripts
python3 build.py     # bundles app.js and worker.js, copies the ONNX Runtime wasm files into ort/
python3 pack.py      # runs starter/export.py --int4 into assets/baseline.onnx
python3 serve.py     # http://127.0.0.1:8765/
```

`pack.py` needs `starter/baseline.pt`. The tests: `npm test` for the page's metadata reader, `python3 -m pytest baseline/test_export.py` for the export.

## Deploy

The page is static. It lives at https://www.sparshith.com/nd-challenge/, served by GitHub Pages from the blog repository.

```sh
python3 deploy.py --to /path/to/blog/nd-challenge    # build, pack, and copy into the chosen site repo
```

Then commit and push `nd-challenge/` in the blog repository. The copy is 24 MB: the page, the two bundles, the wasm runtime (14 MB) and the baseline (9.6 MB). GitHub Pages sends no cross-origin isolation headers, so ONNX Runtime runs single-threaded there: about 40 ms per move for the baseline instead of 30. Build outputs (`*.bundle.js`, `ort/`, `assets/`) are not committed here; `deploy.py` rebuilds them.

## What the page does

1. Reads the vocabulary, context length and name from the file's metadata.
2. Loads the file with ONNX Runtime Web in a worker. Everything runs in the visitor's browser; nothing is uploaded.
3. Runs four game prefixes (0, 7, 31, 63 moves) and checks for finite logits of the right shape.
4. Lets the visitor play the model, or run a 20-game match against the baseline from 10 openings with both colors. The highest-scoring token is selected without legal masking; an illegal or non-move token immediately loses the game.
