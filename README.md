# AI Orchestra

AI Orchestra is a small signal-reconstruction project. The system generates
several clean sine waves, mixes noisy versions into one combined signal, and
trains a neural network to **recover one requested clean sine wave** from a
short window of the noisy mix plus a one-hot vector that names which
frequency the network should produce.

The repository compares three architectures on the exact same data:

- a fully connected feed-forward network (FC)
- a single-layer recurrent network (RNN)
- a single-layer LSTM

The headline result is in the [Results](#results) section: under a proper
held-out-signal-set split, all three models land within a narrow band, and
the RNN advantage that shows up under a leaky random split largely
disappears.

## The Task

For each generated **signal set** the project produces:

- Four clean sine waves at `1 Hz`, `3 Hz`, `5 Hz`, `7 Hz` (configurable),
  each with a random amplitude in `[0.5, 1.5]` and a random phase in
  `[0, 2π)`.
- Four matching **noisy** versions, where each clean sine is perturbed by
  amplitude noise, phase noise, and additive Gaussian noise.
- A single **noisy mix**, the sum of the four noisy sines, which is what
  the model actually sees as input.

For each `(window_start, target_class)` pair the model receives:

- a window of `10` consecutive samples from the noisy mix, and
- a one-hot vector of length `4` that names which clean sine to predict.

It must output the `10`-sample window of the corresponding **clean** sine
at the same time positions.

## The Data

| Concept                | Value (default)                              |
|------------------------|----------------------------------------------|
| Frequencies            | `1`, `3`, `5`, `7` Hz                        |
| Sampling rate          | `1000` Hz                                    |
| Window size            | `10` samples (`10 ms`)                       |
| Signal sets            | `50` by default; smoke runs use `1`–`5`      |
| Amplitude noise std    | `0.10`                                       |
| Phase noise std        | `0.10`                                       |
| Additive noise std     | `0.02`                                       |
| Train/test split       | `80 / 20`                                    |

Signal generation lives in `src/ai_orchestra/signals.py` and
`signal_sets.py`. Two PyTorch-friendly dataset views are available:

- `FullyConnectedSignalDataset` — flat `(14,)` inputs (10 noisy samples
  + 4-value one-hot) and `(10,)` targets, used by the FC model.
- `SequenceSignalDataset` — `(10, 5)` inputs where each time step is one
  noisy sample concatenated with the repeated one-hot, used by the RNN
  and LSTM.

Both datasets are **lazy**: a single index is converted to a
`(set, window, class)` triple on the fly, so a 50-signal-set run never
materializes the full example tensor in memory.

## The Models

| Model | Module                                | Param count        | Input shape |
|-------|---------------------------------------|--------------------|-------------|
| FC    | `FullyConnectedSignalNet` (models.py) | small dense stack  | `(14,)`     |
| RNN   | `RnnSignalNet`                        | `hidden_size=32`   | `(10, 5)`   |
| LSTM  | `LstmSignalNet`                       | `hidden_size=32`   | `(10, 5)`   |

The RNN and LSTM share a common base class `_SequenceSignalNet` so the
sequence-model code path is identical except for the recurrent cell.
Every output goes through a per-time-step linear head that maps the
hidden state to one clean-signal value.

## Setup

Create and activate a local virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The workspace standards recommend `uv` (`pyproject.toml` + `uv.lock`).
The commit history of this project uses `python -m pip` because `uv` was
not available on the development machine; the steps above were what was
actually tested, and `requirements.txt` is kept in sync with the
`pyproject.toml` dependency block.

## Reproducing Results

All entry points expect the package to be importable. From the project
root:

```powershell
$env:PYTHONPATH="src"
```

Then:

```powershell
# 1. Quick FC baseline — prints losses, writes plots under results/figures/fc_baseline/
python -m ai_orchestra

# 2. FC vs RNN vs LSTM head-to-head on 1 signal set (M30 settings)
python scripts/run_comparison.py

# 3. List all experiment variants and run one of them (M32 infrastructure)
python scripts/run_experiment.py --list
python scripts/run_experiment.py noise_low --num-samples 100 --batch-size 32 --signal-set-count 1

# 4. Generalization check: random-window vs held-out-signal-set split (M33)
python scripts/run_generalization_check.py
```

Each entry point writes its artifacts to a different directory under
`results/`, so you can run them in any order without overwriting earlier
output.

## Results

### Headline: random split vs held-out signal set

`scripts/run_generalization_check.py` trains FC / RNN / LSTM under two
splits at identical settings (`5` signal sets, `100` samples,
`batch_size=32`, `3` epochs, `learning_rate=1e-3`, `hidden_size=32`,
`seed=42`, cpu) and compares their overall test MSE:

| Split                     | FC       | RNN      | LSTM     |
|---------------------------|----------|----------|----------|
| Random window split       | `0.4012` | `0.4115` | `0.3999` |
| Held-out signal-set split | `0.5774` | `0.5463` | `0.5550` |
| Delta (holdout − random)  | `+0.1762`| `+0.1349`| `+0.1551`|

The random-window split lets every signal set leak across train and test
(both halves contain *different windows of the same signals*), so it
inflates apparent quality. Once the test signal sets are unseen during
training:

- All three models degrade meaningfully (`+34%` to `+44%` worse MSE).
- The RNN degrades least (`+0.1349`), so its recurrent state seems to
  help a little for truly new signals.
- The three models end up **roughly tied** at `0.55–0.58` MSE. The RNN
  lead seen at smaller scales evaporates under stricter evaluation.

Headline conclusions in this README use the **held-out signal-set
split** for that reason; the random split is kept as a quick smoke
baseline in `compare_models` for fast iteration.

### Smoke comparison at 1 signal set (M30)

`scripts/run_comparison.py` runs all three models on a single signal set
for `5` epochs and writes:

- `results/metrics/model_comparison.json`
- `results/metrics/model_comparison.csv`
- `results/figures/comparison/overall_test_mse.png`
- `results/figures/comparison/per_frequency_test_mse.png`

Per-frequency test MSE (lower is better):

| Model | 1 Hz     | 3 Hz     | 5 Hz     | 7 Hz     | Overall  |
|-------|----------|----------|----------|----------|----------|
| FC    | `0.0613` | `0.6411` | `0.4751` | `0.9012` | `0.5372` |
| RNN   | `0.1092` | `0.1012` | `0.2841` | `0.8820` | `0.3406` |
| LSTM  | `0.0330` | `0.3991` | `0.3134` | `1.1902` | `0.4928` |

- All three models do best at `1 Hz` and worst at `7 Hz`. The fixed
  `10`-sample window covers only `1%` of a `1 Hz` period (smooth and easy)
  versus `7%` of a `7 Hz` period (much harder to fit).
- LSTM gets the single best `1 Hz` result (`0.0330`) and the single
  worst `7 Hz` result (`1.1902`). Its smoothing bias helps on slow
  signals and hurts on fast ones at this scale.
- This run shares signal sets between train and test, so its absolute
  numbers should be read as a *training-fit indicator* rather than a
  generalization claim. The held-out result above is the conservative
  number.

### Plots

Generated when the runners execute:

- `results/figures/fc_baseline/fc_loss_curve.png`,
  `fc_prediction_s{1..4}_{1,3,5,7}hz.png`
- `results/figures/rnn_baseline/rnn_loss_curve.png`,
  `rnn_prediction_s{1..4}_*hz.png`
- `results/figures/lstm_baseline/lstm_loss_curve.png`,
  `lstm_prediction_s{1..4}_*hz.png`
- `results/figures/comparison/overall_test_mse.png`,
  `per_frequency_test_mse.png`
- `results/figures/raw_examples/clean_signals.png`,
  `clean_mixed_signal.png`, `noisy_signals.png`,
  `noisy_mixed_signal.png`

How to read them:

- **Prediction plots** show one clean target window (blue) and the model
  prediction (orange) for one requested frequency. Closer overlap is
  better.
- **Loss curves** show average MSE per epoch on train and test. A
  decreasing curve means the training loop, dataset, optimizer, and
  model wiring are correctly wired together.
- **Overall MSE bar chart** ranks the three models by total test MSE.
- **Per-frequency bar chart** groups bars by target frequency so the
  per-class breakdown is visible.

## Conclusions

- The pipeline works end-to-end: signal generation, conditioning,
  windowing, training, evaluation, plotting, JSON/CSV export, and a
  config-driven experiment runner are all in place.
- All three models *learn the task* in the sense that their loss curves
  decrease monotonically and their predictions are well-behaved (finite,
  bounded, no NaNs).
- Under the **leaky random-window split**, the simple RNN looks meaningfully
  better than FC and LSTM at small scale. Under the **stricter held-out
  signal-set split** that advantage shrinks to about `0.03` MSE, and all
  three models land in a similar band.
- At the smoke scale we used here, **no model produces visually
  near-perfect reconstructions**. The numbers measure trends, not a
  finished product.

## Limitations

- **Scale.** Results in this README come from very small runs
  (`1`–`5` signal sets, `100` samples, `3`–`5` epochs). The defaults file
  is set up for larger runs (`50` sets, `10 000` samples), but no
  large-scale sweep is included; that is left for follow-up work.
- **Short window.** The `10`-sample window at `1000 Hz` covers only
  `10 ms`. That is little temporal structure for any sequence model to
  exploit, which caps how much RNN/LSTM can beat FC without a longer
  window. The `window_large` experiment variant (window `25`) exists
  exactly to probe this, but was not executed in the committed runs.
- **No classical baseline.** A real comparison should also include a
  classical signal-processing baseline (e.g. a tuned band-pass filter at
  each target frequency). That would test whether the neural networks
  beat a non-learned reference at all.
- **CPU only.** All runs in this repository are CPU-bound. There is no
  GPU-specific code path and no benchmarking against one.
- **Reproducibility caveat.** `compare_models` uses one `torch.manual_seed`
  per call; absolute numbers between separate baseline scripts and the
  comparison script differ slightly because the RNG is consumed in a
  different order. Within a single script, runs are deterministic.

## Configuration

Project tunables live in `config/default.json` and mirror the
`PipelineDefaults` dataclass in `src/ai_orchestra/defaults.py`. The JSON
file is the source of truth for the official defaults; the dataclass keeps
matching values as a fallback so importing the package performs no disk
I/O.

```python
from ai_orchestra import load_pipeline_defaults

defaults = load_pipeline_defaults()
defaults = load_pipeline_defaults("config/my_experiment.json")
```

`load_pipeline_defaults()` validates the file: unknown fields, wrong
types, non-positive learning rates, and similar mistakes are rejected
with a clear error. Any custom config must list the same fields as
`config/default.json`.

## Experiment Variants

`config/experiments.json` lists named variants on top of the default
config. Each variant changes one knob (or a closely coupled group like
the three noise standard deviations) so that comparisons stay
interpretable. Available categories:

- noise: `noise_low`, `noise_medium`, `noise_high`
- hidden size: `hidden_small`, `hidden_large`
- recurrent layers: `layers_one`, `layers_two`
- window size: `window_10`, `window_large`
- split: `split_80_20`, `split_70_30`, `split_90_10`

The same flow is available from Python: `run_experiment(name, ...)`,
`load_experiments()`, `apply_overrides(base, overrides)`, and
`list_known_experiments()`. The CLI runner (`scripts/run_experiment.py`)
writes each run's resolved config alongside its metrics so every result
is self-contained.

## Project Layout

```text
src/ai_orchestra/      # SDK: signal gen, datasets, models, training,
                       #      evaluation, comparison, experiments, plotting
tests/unit/            # Unit tests
tests/integration/     # End-to-end smoke tests
config/                # default.json + experiments.json
scripts/               # CLI runners: comparison, experiments, generalization
results/figures/       # Generated PNG plots (committed for review)
results/metrics/       # Generated JSON/CSV metrics
results/experiments/   # Per-experiment artifacts
results/generalization/# Random-vs-holdout split comparison
```

Every Python module is under `150` lines per the project's coding
guidelines; large logical units are split into small, single-purpose
files (for example `comparison.py` plus `comparison_internals.py`,
`comparison_io.py`, and `comparison_plotting.py`).

## Tests And Lint

```powershell
python -m pytest        # full suite
python -m ruff check .  # lint
```

The local quality gate currently passes with **180 tests** and **Ruff
clean**.
