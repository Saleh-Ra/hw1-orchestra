# AI Orchestra

AI Orchestra is a small signal-reconstruction project. It generates several
clean sine waves, combines noisy versions into one mixed signal, and trains a
neural network to reconstruct one requested clean signal from a noisy mixed
window plus a one-hot condition vector.

## Current Pipeline

The implemented baseline is intentionally small:

1. Generate clean sine signals at `1 Hz`, `3 Hz`, `5 Hz`, and `7 Hz`.
2. Randomize amplitudes and phases for each generated signal set.
3. Create noisy versions and sum them into one noisy mixed signal.
4. Build a lazy fully connected dataset from sliding windows.
5. Append a one-hot request vector to each noisy window.
6. Train a simple fully connected model to predict the requested clean window.
7. Save target-vs-prediction plots and a train/test loss curve.

For the fully connected model, each input has shape `(14,)`: ten noisy mixed
samples plus a four-value one-hot condition. Each target has shape `(10,)`.

## Setup

Create and activate a local virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run The Current Baseline

From the project root:

```powershell
$env:PYTHONPATH="src"
python -m ai_orchestra
```

This runs the quick FC baseline, prints train/test losses, and writes plots to:

```text
results/figures/fc_baseline/
```

Current example output from the quick run shows loss decreasing over five
epochs, from about `0.635` to `0.505` on train loss and from about `0.671` to
`0.515` on test loss.

## Results So Far

The current baseline runs end-to-end and produces plots:

- `results/figures/fc_baseline/fc_prediction_s1_1hz.png`
- `results/figures/fc_baseline/fc_prediction_s2_3hz.png`
- `results/figures/fc_baseline/fc_prediction_s3_5hz.png`
- `results/figures/fc_baseline/fc_prediction_s4_7hz.png`
- `results/figures/fc_baseline/fc_loss_curve.png`

The prediction plots compare one requested clean target window against the FC
model output. The blue line is the true clean signal window, and the orange line
is the model prediction. A good reconstruction should have the orange line
closely overlap the blue line for each requested frequency.

The loss curve shows average mean squared error across epochs. Both train and
test loss decrease in the current quick run, which means the training loop,
dataset, optimizer, and model wiring are working. Loss decreasing alone is not
enough, though: the prediction plots still show the FC model is not yet
reconstructing the requested windows well.

The current runner is a quick smoke baseline: one generated signal set, `100`
samples, and `5` epochs. It is useful for proving the code path works, but not
for final model quality. Before adding RNN and LSTM models, the next decision is
whether to improve the FC baseline with a stronger run, better normalization,
more training data, or adjusted model/training settings.

## Tests And Lint

Run the full test suite:

```powershell
python -m pytest
```

Run Ruff:

```powershell
python -m ruff check .
```

At this point, the local quality gate passes with `115` tests.
