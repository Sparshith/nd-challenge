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

`pack.py` downloads `starter/baseline.pt` on the first run. The tests: `npm test` for browser logic, `python3 -m pytest` from the repository root for entry rules and export parity.

## Deploy

GitHub Actions builds, tests, and publishes this repository to https://www.sparshith.com/nd-challenge/ on every push to `main`. Pull requests build and test without publishing. No files are copied into the blog repository.

```sh
python3 dashboard/package_site.py    # from the repository root, build output/site
```

The artifact contains only the page, styles, browser bundles, ONNX Runtime files, and baseline. Generated files stay untracked. GitHub Pages runs ONNX Runtime single-threaded because it does not provide cross-origin isolation headers.

Repository Settings → Pages must use **GitHub Actions**. Leave the custom domain empty: this project inherits `www.sparshith.com` from the account's main site, and the repository name `nd-challenge` supplies the URL path. The deployment job is restricted to `Sparshith/nd-challenge` on `main`; forks can run the checks without publishing to the official site.

## What the page does

1. Reads the vocabulary, context length and name from the file's metadata.
2. Loads the file with ONNX Runtime Web in a worker. Everything runs in the visitor's browser; nothing is uploaded.
3. Runs four game prefixes (0, 7, 31, 63 moves) and checks for finite logits of the right shape.
4. Lets the visitor play the model, or run a 20-game match against the baseline from 10 openings with both colors. The highest-scoring token is selected without legal masking; an illegal or non-move token immediately loses the game.
