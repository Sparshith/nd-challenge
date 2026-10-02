# nanoDanya challenge

How much chess can a 10 MB language model learn?

Your model reads the moves played so far and predicts the next move. It gets no
board position or list of legal moves. **An illegal prediction immediately loses
the game.** No search or engine assistance at inference time.

The model must be a **causal next-token decoder**, exported as one ONNX file of at
most **10,000,000 bytes**. Board-input models such as the Searchless action-value
model are outside this challenge's scope. Training is unrestricted, including
engine labels, pretrained weights, distillation, and RL.

The starter is the current baseline: **1303 ±16 internal Elo, 22.1% puzzles
solved, 9.45 MB** without legal-move masking. [Evaluation](baseline/evaluation.md).

## Start from a fork

Fork this repository, clone your fork, and run:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r starter/requirements.txt
cd starter
python3 export.py --int4
python3 check.py --games 2
```

The first export downloads the baseline weights (8.9 MB). These commands do not
train or download the training dataset. Once the baseline works, modify
`starter/model.py`, train with `starter/train.py`, and export your checkpoint:

```sh
python3 export.py --checkpoint best.pt --name "My model" --int4
python3 check.py
```

See the [starter guide](starter/README.md) and [full rules](llms.txt).
The [browser checker](https://www.sparshith.com/nd-challenge/) is built and deployed
from this repository. Pull requests run the tests and build; merges into `main`
automatically publish the site after those checks pass.

## Submit

Open a [model submission issue](https://github.com/Sparshith/nd-challenge/issues/new/choose) with:

- A public, direct-download URL for the exact `model.onnx` file, its byte size and SHA-256.
- A link to your fork at a fixed commit containing the model, training and export code, pinned dependencies, and export instructions.
- The final checkpoint needed to reproduce the export, as a release/download link rather than a large Git blob.
- The complete checker report and a short description of the architecture, training data, and compression.

**Passing the checker confirms compatibility. Leaderboard entry requires source
review and official evaluation.** We review the forward pass and export code,
reproduce the artifact, and run the submitted ONNX ourselves. Participants do not
supply a game-playing wrapper. Our evaluator selects the raw highest-scoring
token; invalid moves and non-move tokens lose immediately.

Official evaluation uses 8,800 puzzles and 1,200 games from 200 openings, both
colors, against three Stockfish settings. Masked and unmasked results use separate
rating pools. Submission issues are a manual queue and do not launch compute.
After official evaluation, maintainers merge approved leaderboard updates into
`dashboard/index.html`; that merge publishes the updated scores. Replacing the
downloadable baseline also requires updating its checkpoint/export recipe and
evaluation results in the same change.

## Repository layout

- `starter/`: model, training, export, tokenizer, vocabulary, and local checker.
- `dashboard/`: browser checker and game interface, including its public openings.
- `baseline/`: baseline results and export parity test.
- `tests/`: entry and move-selection checks.
- `llms.txt`: complete entry contract and coding-agent setup instructions.

Run Python tests with `pip install -r requirements-dev.txt` and
`python3 -m pytest`. For the browser source, see [dashboard setup](dashboard/README.md).
The research experiments remain in [nanoDanya](https://github.com/Sparshith/nanoDanya).

Baseline weights remain hosted on the existing `nanoDanya` release, and the
training data remains on Hugging Face. Forking this repository does not copy or
change either artifact.
