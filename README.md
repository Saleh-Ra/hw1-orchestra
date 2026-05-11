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

Three baselines run end-to-end today: fully connected (FC), a simple
recurrent network (RNN), and an LSTM. All three train on the same signal
sets with the same split seed; the RNN and LSTM consume a sequence-format
view of the same data where each time step carries one noisy sample and the
repeated one-hot condition.

### Plots

- `results/figures/fc_baseline/fc_prediction_s{1..4}_*hz.png`
- `results/figures/fc_baseline/fc_loss_curve.png`
- `results/figures/rnn_baseline/rnn_prediction_s{1..4}_*hz.png`
- `results/figures/rnn_baseline/rnn_loss_curve.png`
- `results/figures/lstm_baseline/lstm_prediction_s{1..4}_*hz.png`
- `results/figures/lstm_baseline/lstm_loss_curve.png`

Each prediction plot compares one requested clean target window against the
model output. The blue line is the true clean signal window, and the orange
line is the model prediction. A good reconstruction should have the orange
line closely overlap the blue line for each requested frequency.

Each loss curve shows average mean squared error across epochs. Both train
and test loss decrease in the current quick runs, which means the training
loop, dataset, optimizer, and model wiring are working for both models.

### FC vs RNN vs LSTM Comparison

Same-seed, same-defaults quick run (1 signal set, 100 samples, batch 32,
5 epochs, learning rate `1e-3`, hidden size `32`):

| Model | Final train loss | Final test loss |
| ----- | ---------------- | --------------- |
| FC    | `0.5049`         | `0.5148`        |
| RNN   | `0.3323`         | `0.3337`        |
| LSTM  | `0.5369`         | `0.5636`        |

At this smoke-run scale the simple RNN clearly leads: about `34%` lower
train loss and `35%` lower test loss than FC, with a near-zero train/test
gap and no instability. The LSTM is **undertrained** here — five epochs and
one signal set is too little for its extra gate parameters to converge, so
it lands close to FC and behind RNN. LSTM loss is still decreasing per
epoch, finite, and stable; it just has not caught up yet.

### Honest Limitations

The current runner is a quick smoke baseline: one signal set, `100` samples,
`5` epochs. Loss decreases are real, but the prediction plots still show
the models do not yet closely reconstruct the requested windows.

The window size is also short relative to the signal periods: `10` samples
at `1000 Hz` cover `10 ms`, which is only `1%` of one period at `1 Hz` and
about `7%` at `7 Hz`. That is very little temporal structure for any sequence
model to exploit, which caps how much RNN or LSTM can beat FC without
growing the window or the data scale. A proper experiment milestone with
more signal sets and more epochs is needed before judging whether LSTM
eventually surpasses RNN on this task.

## Tests And Lint

Run the full test suite:

```powershell
python -m pytest
```

Run Ruff:

```powershell
python -m ruff check .
```

At this point, the local quality gate passes with `139` tests.
