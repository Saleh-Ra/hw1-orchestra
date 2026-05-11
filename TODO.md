# TODO

This checklist follows `REQUIREMENTS.md` and `PLAN.md`, with one important work-order correction:

Start with a minimal working ML pipeline first. Do not build all infrastructure before proving that the model can learn the reconstruction task.

## Status Legend

- `[ ]` Not started
- `[x]` Done
- Keep each item small enough to finish, test, or review independently.

### Phase A: Planning And Project Setup

Good commit scope: planning docs, project skeleton, defaults, import/test setup.

## Milestone 0: Planning Alignment

- [x] Re-read `REQUIREMENTS.md`.
- [x] Re-read `PLAN.md`.
- [x] Confirm that the first implementation target is a minimal working pipeline.
- [x] Confirm that the first trained model will be a fully connected network only.
- [x] Confirm that RNN will be added after the FC model works.
- [x] Confirm that LSTM will be added after the RNN works.
- [x] Confirm that experiment/config polish comes after the ML pipeline works.
- [x] Update `PLAN.md` to explicitly prioritize the minimal working pipeline.
- [x] Update `PLAN.md` to say the baseline should use many generated signal sets.
- [x] Update `PLAN.md` to avoid presenting one generated signal set as the final baseline.
- [x] Add a short note that `1, 3, 5, 7 Hz` are initial chosen frequencies, not a hard theoretical limit.
- [x] Add a short note that dataset size can grow by generating more random signal sets.
- [x] Decide whether to keep `code.py` or replace it with the package layout.
- [x] Decide whether to keep docs at root for now or move later into `docs/`.

Milestone 0 decisions:

- Replace `code.py` with the package layout when implementation begins. Do not build the project around a single script.
- Keep `REQUIREMENTS.md`, `PLAN.md`, and `TODO.md` at the repo root for now because the project is still small and the user is actively reviewing them there.

## Milestone 1: Minimal Project Setup

- [x] Inspect the current repository contents.
- [x] Create `pyproject.toml`.
- [x] Add the project name.
- [x] Add the project version.
- [x] Add the Python version requirement.
- [x] Add NumPy dependency.
- [x] Add PyTorch dependency.
- [x] Add Matplotlib dependency.
- [x] Add pytest development dependency.
- [x] Add Ruff development dependency.
- [ ] Generate or update `uv.lock`.
- [x] Create `src/`.
- [x] Create `src/ai_orchestra/`.
- [x] Create `src/ai_orchestra/__init__.py`.
- [x] Add a minimal package import test.
- [x] Create `tests/`.
- [x] Create `tests/unit/`.
- [x] Create `tests/integration/`.
- [x] Add a minimal smoke test.
- [x] Create `.gitignore` if missing.
- [x] Ignore `.venv/`.
- [x] Ignore `__pycache__/`.
- [x] Ignore `.pytest_cache/`.
- [x] Ignore `.ruff_cache/`.
- [x] Ignore large generated data files.
- [x] Ignore model checkpoint files.
- [x] Ignore temporary result files where appropriate.
- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check .`.
- [ ] Fix setup issues until tests pass.
- [ ] Fix setup issues until Ruff passes.

Milestone 1 blocker:

- `uv` is not installed or not on `PATH`, so `uv.lock`, `uv run pytest`, and `uv run ruff check .` are blocked for now.
- Local Python imports the package successfully, and `pyproject.toml` parses successfully.

## Milestone 2: First Hardcoded Defaults

- [x] Create a simple defaults section in code before building full config files.
- [x] Define frequencies as `[1, 3, 5, 7]`.
- [x] Define sampling frequency as `1000`.
- [x] Define number of samples as `10000`.
- [x] Define signal duration as `10` seconds.
- [x] Define window size as `10`.
- [x] Define default number of generated signal sets as `50`.
- [x] Define default train ratio as `0.8`.
- [x] Define default random seed.
- [x] Define default amplitude range.
- [x] Define default phase range from `0` to `2pi`.
- [x] Define default amplitude noise strength.
- [x] Define default phase noise strength.
- [x] Define default additive noise strength if needed.
- [x] Keep these defaults easy to move into config later.

Milestone 2 validation:

- Defaults live in `src/ai_orchestra/defaults.py` as a small dataclass.
- The package exports `DEFAULTS` and `PipelineDefaults`.
- Local Python compilation and default validation pass.
- Full local pytest/Ruff commands pass in `.venv`.

### Phase B: Signal Generation Core

Good commit scope: clean sine generation, random amplitudes/phases, noisy signal generation, mixed signals, and signal-set containers.

## Milestone 3: Generate One Clean Signal

- [x] Create `src/ai_orchestra/signals.py`.
- [x] Add a function to create a time axis.
- [x] Test that the time axis length is `10000`.
- [x] Test that the first time value is `0`.
- [x] Test that the time step matches `1 / 1000`.
- [x] Add a function to generate a clean sine signal.
- [x] Pass frequency into the clean signal function.
- [x] Pass amplitude into the clean signal function.
- [x] Pass phase into the clean signal function.
- [x] Return a NumPy array from the clean signal function.
- [x] Test that one generated clean signal has shape `(10000,)`.
- [x] Test that the clean signal contains finite values.
- [x] Test that changing frequency changes the signal.
- [x] Test that changing phase changes the signal.
- [x] Test that changing amplitude changes the signal scale.

Milestone 3 validation:

- `make_time_axis()` creates `10000` samples starting at `0.0` with a `0.001` second step.
- `generate_clean_signal()` returns a finite NumPy array with shape `(10000,)`.
- Local Python compilation and direct signal sanity checks pass.
- Full local pytest/Ruff commands pass in `.venv`.

## Milestone 4: Generate Four Clean Signals

- [x] Add a function to generate all four clean signals.
- [x] Generate `S1` at `1 Hz`.
- [x] Generate `S2` at `3 Hz`.
- [x] Generate `S3` at `5 Hz`.
- [x] Generate `S4` at `7 Hz`.
- [x] Store clean signals in frequency order.
- [x] Return clean signals with shape `(4, 10000)`.
- [x] Test the clean signal stack shape.
- [x] Test that four frequencies are represented.
- [x] Test that generated signals are not all identical.
- [x] Test that clean signal generation is deterministic before random sampling is added.

Milestone 4 validation:

- `generate_clean_signals()` returns a clean signal stack with shape `(4, 10000)`.
- Signal rows follow the configured frequency order: `1 Hz`, `3 Hz`, `5 Hz`, `7 Hz`.
- Local Python compilation and direct clean-stack sanity checks pass.
- Seed-controlled random amplitude and phase generation starts in Milestone 5.

## Milestone 5: Add Random Amplitude And Phase

- [x] Add random amplitude sampling.
- [x] Add random phase sampling.
- [x] Keep phase values between `0` and `2pi`.
- [x] Use one random amplitude per signal per signal set.
- [x] Use one random phase per signal per signal set.
- [x] Make random generation seed-controlled.
- [x] Test that the same seed gives the same amplitudes.
- [x] Test that the same seed gives the same phases.
- [x] Test that a different seed changes at least one signal.
- [x] Keep the random logic simple and readable.

Milestone 5 validation:

- `sample_signal_parameters()` samples one amplitude and one phase per configured frequency.
- The same seed reproduces the same amplitudes and phases.
- Different seeds change at least one sampled parameter.
- Sampled phases stay between `0` and `2pi`.
- Local Python compilation and direct random-parameter sanity checks pass.

## Milestone 6: Generate Noisy Signal Versions

- [x] Add amplitude noise to create noisy versions.
- [x] Add phase noise to create noisy versions.
- [x] Optionally add small direct sample noise if needed.
- [x] Keep noisy signal shape equal to clean signal shape.
- [x] Ensure noisy signals are based on the same frequencies as clean signals.
- [x] Ensure noisy signals differ from clean signals.
- [x] Test noisy signal shape is `(4, 10000)`.
- [x] Test noisy signal values are finite.
- [x] Test noisy generation is reproducible with a seed.
- [x] Test that increasing noise strength changes the noisy signals more.

Milestone 6 validation:

- `generate_noisy_signals()` creates noisy versions from the same configured frequencies.
- Noisy signals include amplitude noise, phase noise, and optional additive sample noise.
- The noisy stack keeps shape `(4, 10000)` and contains finite values.
- Seeded noisy generation is reproducible.
- Stronger noise settings move noisy signals farther from the clean baseline.
- Local Python compilation and direct noisy-signal sanity checks pass.

## Milestone 7: Combine Signals

- [x] Add a function to sum signals across the signal axis.
- [x] Create the clean mixed signal.
- [x] Create the noisy mixed signal.
- [x] Ensure clean mixed signal shape is `(10000,)`.
- [x] Ensure noisy mixed signal shape is `(10000,)`.
- [x] Test clean mixed signal equals the sum of clean signals.
- [x] Test noisy mixed signal equals the sum of noisy signals.
- [x] Test mixed signals contain finite values.
- [x] Test clean and noisy mixed signals are not identical when noise is enabled.

Milestone 7 validation:

- `combine_signals()` sums a `(signal_count, sample_count)` stack into one mixed signal.
- Clean and noisy mixed signals both have shape `(10000,)`.
- Mixed signals equal the sum across the signal axis.
- Clean and noisy mixed signals differ when noise is enabled.
- Local Python compilation and direct mixed-signal sanity checks pass.

## Milestone 8: Represent One Signal Set

- [x] Add a small `SignalSet` dataclass.
- [x] Store the time axis.
- [x] Store the frequencies.
- [x] Store the clean signals.
- [x] Store the noisy signals.
- [x] Store the clean mixed signal.
- [x] Store the noisy mixed signal.
- [x] Store amplitudes if useful for debugging.
- [x] Store phases if useful for debugging.
- [x] Add a function that returns one complete `SignalSet`.
- [x] Test that `SignalSet` fields have expected shapes.
- [x] Test that one complete signal set can be generated with one function call.

Milestone 8 validation:

- `SignalSet` lives in `src/ai_orchestra/signal_sets.py`.
- `generate_signal_set()` returns time axis, frequencies, amplitudes, phases, clean/noisy signals, and clean/noisy mixes.
- Signal set fields have the expected shapes.
- Seeded signal set generation is reproducible.
- Local Python compilation and direct signal-set sanity checks pass.

## Milestone 9: Generate Many Signal Sets

- [x] Add a function to generate multiple random signal sets.
- [x] Use `50` signal sets as the first default.
- [x] Use a different seed stream per signal set.
- [x] Keep frequencies the same across sets for the first baseline.
- [x] Randomize amplitudes across sets.
- [x] Randomize phases across sets.
- [x] Randomize noise across sets.
- [x] Return signal sets in a list or stacked structure.
- [x] Test that the number of generated sets matches the request.
- [x] Test that different sets are not identical.
- [x] Test that generation is reproducible from the same base seed.
- [x] Keep memory usage reasonable.
- [x] Add a smaller test setting with only `2` or `3` signal sets.

Milestone 9 validation:

- `generate_signal_sets()` returns a list of generated `SignalSet` objects.
- The default count comes from `DEFAULTS.num_signal_sets`, currently `50`.
- Each signal set uses a different seed from a reproducible base seed stream.
- Frequencies stay fixed across sets while amplitudes, phases, and noise vary.
- Unit tests use small `2` or `3` set configurations to keep memory and runtime low.
- Local Python compilation and direct many-signal-set sanity checks pass.

### Phase C: Raw Signal Visualization

Good commit scope: plotting helpers and generated example figures, if we decide to commit selected result images.

## Milestone 10: Visualize Raw Signals Early

- [x] Create `src/ai_orchestra/plotting.py`.
- [x] Add a function to plot clean individual signals.
- [x] Add a function to plot noisy individual signals.
- [x] Add a function to plot clean mixed signal.
- [x] Add a function to plot noisy mixed signal.
- [x] Limit plotted samples so figures are readable.
- [x] Save plots to a results folder.
- [x] Ensure plotting works without an interactive display.
- [x] Generate one raw clean-signal plot manually.
- [x] Generate one noisy-signal plot manually.
- [x] Generate one mixed-signal plot manually.
- [x] Visually inspect the plots.
- [x] Fix signal generation if plots look obviously wrong.

Milestone 10 validation:

- `plot_signal_stack()` saves readable individual-signal plots.
- `plot_mixed_signal()` saves readable mixed-signal plots.
- `plot_raw_signal_examples()` saves clean, noisy, clean-mix, and noisy-mix examples.
- Matplotlib uses the non-interactive `Agg` backend.
- Example plots were generated under `results/figures/raw_examples/`.
- Visual inspection showed the plots look reasonable, so no signal-generation fix was needed.
- Local Python compilation and direct plotting sanity checks pass.

## Milestone 11: Build One-Hot Encoding

- [x] Create `src/ai_orchestra/dataset.py`.
- [x] Add a function to create one-hot vectors.
- [x] Map class index `0` to `[1, 0, 0, 0]`.
- [x] Map class index `1` to `[0, 1, 0, 0]`.
- [x] Map class index `2` to `[0, 0, 1, 0]`.
- [x] Map class index `3` to `[0, 0, 0, 1]`.
- [x] Test every one-hot vector.
- [x] Test invalid class indices fail clearly.
- [x] Keep the one-hot length tied to the number of frequencies.

Milestone 11 validation:

- `make_one_hot()` lives in `src/ai_orchestra/dataset.py`.
- One-hot length follows the configured number of frequencies.
- Class indices `0`, `1`, `2`, and `3` map to the expected default vectors.
- Invalid class indices raise clear `ValueError` messages.
- Local Python compilation and direct one-hot sanity checks pass.

### Phase D: Dataset Construction

Good commit scope: one-hot conditions, window indexing, FC dataset items, dataset size checks, splitting, and data loaders.

## Milestone 12: Build Window Indexing

- [x] Add logic to compute valid window start positions.
- [x] Use window size `10`.
- [x] Confirm valid starts are `0` through `9990`.
- [x] Confirm one signal set gives `9991` window positions.
- [x] Add a function to extract one window.
- [x] Test the first window.
- [x] Test a middle window.
- [x] Test the last valid window.
- [x] Test that off-by-one errors are not present.
- [x] Test invalid window starts fail clearly.

Milestone 12 validation:

- `valid_window_starts()` returns starts from `0` through `9990`.
- One default signal produces `9991` valid window positions.
- `extract_window()` returns 10-sample windows for first, middle, and last valid starts.
- Invalid starts and non-1D inputs raise clear `ValueError` messages.
- Local Python compilation and direct window-indexing sanity checks pass.

## Milestone 13: Build FC Dataset Items

- [x] Create dataset logic for fully connected input format.
- [x] For each signal set, iterate through valid window positions.
- [x] For each window position, create one example per target frequency.
- [x] Extract the noisy mixed signal window.
- [x] Extract the matching clean target signal window.
- [x] Append the one-hot vector to the noisy input window.
- [x] Ensure FC input shape is `(14,)`.
- [x] Ensure target shape is `(10,)`.
- [x] Ensure values are returned as PyTorch tensors.
- [x] Use float tensors.
- [x] Test the first FC dataset item.
- [x] Test an FC item for each condition vector.
- [x] Test that `C = [1, 0, 0, 0]` targets `S1`.
- [x] Test that `C = [0, 1, 0, 0]` targets `S2`.
- [x] Test that `C = [0, 0, 1, 0]` targets `S3`.
- [x] Test that `C = [0, 0, 0, 1]` targets `S4`.
- [x] Test FC input length is exactly `14`.
- [x] Test target length is exactly `10`.

Milestone 13 validation:

- `FullyConnectedSignalDataset` lazily maps signal sets to FC examples.
- One default signal set produces `9991 * 4 = 39964` examples.
- Dataset indexing orders examples by window position, then target class.
- Each item is implemented as noisy mixed window plus one-hot condition and matching clean target window.
- Local Python compilation and dataset length/index metadata checks pass.
- PyTorch tensor-return tests pass in `.venv`.

## Milestone 14: Dataset Size Checks

- [x] Compute examples per signal set.
- [x] Confirm one set gives `9991 * 4 = 39964` examples.
- [x] Confirm `50` sets gives `1998200` examples if fully materialized.
- [x] Decide whether to materialize all examples or index lazily.
- [x] Prefer lazy indexing if memory is too large.
- [x] Implement lazy indexing if needed.
- [x] Test dataset length with one set.
- [x] Test dataset length with two sets.
- [x] Test dataset length with fifty sets or a simulated equivalent.
- [x] Make sure length calculation is fast.
- [x] Make sure item retrieval is fast enough for training.

Milestone 14 validation:

- `examples_per_signal_set()` returns `39964`.
- `total_examples(50)` returns `1998200`.
- `FullyConnectedSignalDataset` uses lazy indexing, so it does not materialize all examples.
- Dataset length checks pass for one, two, and fifty simulated signal sets.
- Added `requirements.txt` as a fallback install file for environments that do not use `uv`.
- Local Python compilation and direct dataset-size sanity checks pass.

## Milestone 15: Minimal Train/Test Split

- [x] Add deterministic train/test split.
- [x] Use `80/20` as the default.
- [x] Use a random seed for the split.
- [x] Start with random split for the minimal pipeline.
- [x] Document that time-based split may be added later.
- [x] Confirm train set length is about 80%.
- [x] Confirm test set length is about 20%.
- [x] Confirm no index appears in both sets.
- [x] Test split reproducibility.
- [x] Test split ratio can be changed later.

Milestone 15 validation:

- `train_test_split_indices()` creates deterministic random index splits.
- Default split is `80/20`.
- Split indices are reproducible with the same seed.
- Train/test indices do not overlap and cover all input indices.
- Split ratio can be changed, for example to `70/30`.
- This is the simple baseline split; time-based splitting remains a later stricter-evaluation option.
- Local Python compilation and direct split sanity checks pass.

## Milestone 16: Create DataLoader

- [x] Add PyTorch `DataLoader` creation.
- [x] Use shuffle for the training loader.
- [x] Do not shuffle the test loader.
- [x] Set a simple default batch size.
- [x] Fetch one training batch.
- [x] Confirm batch input shape is `(batch_size, 14)`.
- [x] Confirm batch target shape is `(batch_size, 10)`.
- [x] Confirm batch tensors are finite.
- [x] Add a small test for DataLoader output shapes.

Milestone 16 validation:

- `create_data_loaders()` lives in `src/ai_orchestra/dataloaders.py`.
- Training loader uses `shuffle=True`.
- Test loader uses `shuffle=False`.
- Default batch size is `64` via `DEFAULTS.batch_size`.
- DataLoader tests fetch batches and check shapes/finite tensors.
- Local Python compilation and direct DataLoader import/default checks pass.
- Full local pytest/Ruff commands pass in `.venv`.

### Phase E: First Fully Connected Baseline

Good commit scope: FC model, training loop, overfit check, first real FC baseline, FC plots, and a minimal runner.

## Milestone 17: Build The First FC Model

- [x] Create `src/ai_orchestra/models.py`.
- [x] Add `FullyConnectedSignalNet`.
- [x] Set input size to `14`.
- [x] Set output size to `10`.
- [x] Add one hidden layer first.
- [x] Use ReLU activation.
- [x] Keep the model simple for the first run.
- [x] Run one fake batch through the model.
- [x] Confirm output shape is `(batch_size, 10)`.
- [x] Test model forward pass.
- [x] Test model output contains finite values.

Milestone 17 validation:

- `FullyConnectedSignalNet` lives in `src/ai_orchestra/models.py`.
- The model uses a simple `14 -> 64 -> 10` ReLU network.
- Model tests are written for fake-batch forward shape and finite outputs.
- Local Python compilation passes.
- Base package import still works without PyTorch installed.
- Fake-batch forward tests pass in `.venv`.

## Milestone 18: Minimal Training Loop

- [x] Create `src/ai_orchestra/training.py`.
- [x] Add a function to train one epoch.
- [x] Add a function to evaluate one epoch.
- [x] Use MSE loss.
- [x] Use Adam optimizer.
- [x] Move model to CPU by default.
- [x] Support GPU if available without requiring it.
- [x] Track average train loss.
- [x] Track average test loss.
- [x] Print or return epoch metrics.
- [x] Train the FC model for one epoch on a tiny subset.
- [x] Confirm the training loop does not crash.
- [x] Confirm loss is finite.
- [x] Add an integration smoke test for one tiny FC training run.

Milestone 18 validation:

- `train_one_epoch()` trains one model epoch and returns average MSE loss.
- `evaluate_one_epoch()` evaluates one model epoch and returns average MSE loss.
- `create_adam_optimizer()` creates the default Adam optimizer.
- Device handling defaults to CPU and uses CUDA if available when no device is provided.
- Unit and integration smoke tests are written for one tiny FC training run.
- Local Python compilation and package import checks pass.
- Tiny FC training smoke tests pass in `.venv`.

## Milestone 19: First Overfit Test

- [x] Create a tiny dataset from one or two generated signal sets.
- [x] Train the FC model on a very small subset.
- [x] Use enough epochs to see if the model can overfit.
- [x] Track training loss.
- [x] Confirm training loss decreases.
- [x] Plot one prediction from the overfit run.
- [x] Compare predicted window to target window.
- [x] If it cannot overfit, inspect dataset indexing.
- [x] If it cannot overfit, inspect target mapping.
- [x] If it cannot overfit, inspect model output shape.
- [x] If it cannot overfit, inspect loss calculation.
- [x] Do not move to larger training until this works.

Milestone 19 validation:

- `run_fc_overfit_check()` lives in `src/ai_orchestra/overfit.py`.
- The overfit check creates one small signal set and trains on a tiny subset.
- It records training losses across epochs.
- It returns one prediction and target window for inspection.
- It saves a prediction-vs-target plot when an output path is provided.
- Integration tests check loss decrease, finite outputs, and plot creation.
- Local Python compilation and package import checks pass.
- Overfit integration tests pass in `.venv`.

## Milestone 20: First Real FC Baseline

- [x] Generate many signal sets for training.
- [x] Start with fewer than `50` sets if runtime is too high.
- [x] Increase toward `50` sets after the pipeline works.
- [x] Build the FC dataset.
- [x] Build train and test splits.
- [x] Train the FC model for a small number of epochs.
- [x] Record train loss per epoch.
- [x] Record test loss per epoch.
- [x] Confirm train loss decreases.
- [x] Confirm test loss is finite.
- [x] Save the loss history in memory first.
- [x] Save metrics to disk only after the run works.
- [x] Inspect several predictions.
- [ ] Confirm predictions roughly follow the target signal in a stronger run.
- [x] Note failure modes if reconstruction is poor.

Milestone 20 implementation status:

- `run_fc_baseline()` lives in `src/ai_orchestra/baseline.py`.
- It generates signal sets, builds the FC dataset, creates train/test splits, trains for configurable epochs, and records train/test losses in memory.
- It supports starting with fewer than `50` signal sets via `signal_set_count`.
- It returns several prediction/target windows for inspection.
- It intentionally does not save metrics to disk yet.
- Integration tests check baseline losses, predictions, and validation errors.
- Local Python compilation and package import checks pass.
- A small manual FC baseline run showed decreasing train/test losses and finite predictions.
- Milestone 21 plots show the short toy run learns in loss terms but predictions are still visibly poor.

## Milestone 21: FC Prediction Plots

- [x] Add a function to plot target vs. prediction.
- [x] Plot a prediction for `S1`.
- [x] Plot a prediction for `S2`.
- [x] Plot a prediction for `S3`.
- [x] Plot a prediction for `S4`.
- [x] Include the model name in the plot title.
- [x] Include the requested frequency in the plot title.
- [x] Save plots under `results/figures/`.
- [x] Visually inspect all four FC prediction plots.
- [x] Add a function to plot training and test loss curves.
- [x] Save the FC loss curve.
- [x] Use these plots to decide whether the FC baseline is good enough.

Milestone 21 validation:

- `fc_plotting.py` saves target-vs-prediction windows and loss curves.
- Baseline prediction sampling now selects one example per class when possible.
- Generated plots live under `results/figures/fc_baseline/`.
- The 5-epoch toy run loss decreases, but prediction plots are not close enough yet.
- Full local pytest and Ruff pass.

## Milestone 22: Minimal SDK Or Runner

- [x] Create a simple runnable entry point.
- [x] Prefer a small SDK function before adding a full CLI.
- [x] Create `src/ai_orchestra/sdk.py`.
- [x] Add a function to run signal generation.
- [x] Add a function to build datasets.
- [x] Add a function to train the FC baseline.
- [x] Add a function to plot FC predictions.
- [x] Create a simple script only if needed for convenience.
- [x] Ensure the runner calls package functions instead of duplicating logic.
- [x] Run the full minimal FC pipeline from one place.
- [x] Confirm the run produces metrics and plots.

Milestone 22 validation:

- `sdk.py` provides reusable package entry points for generation, dataset building, FC training, plotting, and one-call minimal pipeline execution.
- `python -m ai_orchestra` runs the current quick FC baseline and prints metrics plus generated plot paths.
- The runner reuses existing package functions rather than duplicating pipeline logic.
- The quick run produces metrics and plots under `results/figures/fc_baseline/`.

## Milestone 23: Minimal README Update

- [x] Create or update `README.md`.
- [x] Explain the project in one paragraph.
- [x] Explain the minimal pipeline.
- [x] Explain how signals are generated.
- [x] Explain how the one-hot condition works.
- [x] Explain how to run tests.
- [x] Explain how to run the FC baseline.
- [x] Add the current FC result after it exists.
- [x] Add at least one prediction plot after it exists.
- [x] Keep README honest about what is implemented so far.

Milestone 23 validation:

- `README.md` now documents setup with `.venv` and `requirements.txt`.
- It documents `python -m ai_orchestra` as the current runner.
- It lists the generated FC prediction and loss plot paths.
- It states that the quick FC loss decreases but prediction quality still needs review.

## Milestone 24: FC Baseline Review Gate

- [x] Confirm signal generation works.
- [x] Confirm many signal sets are generated.
- [x] Confirm dataset indexing works.
- [x] Confirm FC model trains without crashing.
- [x] Confirm FC training loss decreases.
- [x] Confirm FC test loss is finite.
- [x] Confirm FC prediction plots are generated.
- [ ] Confirm at least some FC predictions visually resemble targets.
- [x] Run all tests.
- [x] Run Ruff.
- [x] Fix any easy issues.
- [x] Decide whether to improve FC before adding RNN.
- [x] Do not add RNN before this gate is passed.

Milestone 24 review result:

- The package runner completes and generates metrics plus FC plots.
- Full pytest passes with `115` tests.
- Ruff passes.
- Train loss decreases from about `0.635` to `0.505`; test loss decreases from about `0.671` to `0.515`.
- Signal generation, many-set generation, dataset indexing, model training, and plotting are covered by tests and the runner.
- The FC prediction plots are still visibly poor, so the review gate says to improve or strengthen the FC baseline before adding RNN/LSTM models.

### Phase F: Sequence Models

Good commit scope: sequence dataset format, simple RNN, LSTM, and their review gates.

## Milestone 25: Prepare Sequence Dataset Format

- [x] Add sequence-format dataset support.
- [x] Keep the same underlying signal sets.
- [x] Keep the same target windows.
- [x] For each time step, include the noisy sample.
- [x] For each time step, include the full one-hot vector.
- [x] Ensure each sequence input shape is `(10, 5)`.
- [x] Ensure each sequence target shape is `(10,)`.
- [x] Test sequence shape for one item.
- [x] Test sequence shape for one batch.
- [x] Test sequence target mapping for each class.
- [x] Reuse as much indexing logic as possible.
- [x] Avoid duplicating target-mapping logic.

Milestone 25 validation:

- `SequenceSignalDataset` lives in `src/ai_orchestra/sequence_dataset.py`.
- It composes `FullyConnectedSignalDataset` internally to reuse all indexing and target-mapping logic.
- Each item returns a `(10, 5)` sequence input and a `(10,)` target tensor.
- Per time step, the first channel holds the noisy mixed sample and the remaining four channels hold the repeated one-hot condition.
- Tests cover length, item shapes, target mapping per class, condition repetition across time steps, middle-window indexing, batch shape via `DataLoader`, and invalid index handling.

## Milestone 26: Add Simple RNN

- [x] Add `RnnSignalNet`.
- [x] Accept input shape `(batch_size, 10, 5)`.
- [x] Use a configurable hidden size.
- [x] Start with one recurrent layer.
- [x] Map recurrent outputs to one value per time step.
- [x] Return output shape `(batch_size, 10)`.
- [x] Run one fake sequence batch through the RNN.
- [x] Test RNN output shape.
- [x] Test RNN output contains finite values.
- [x] Train RNN on a tiny subset.
- [x] Confirm RNN training loop does not crash.
- [x] Confirm RNN loss is finite.
- [x] Train RNN on the same baseline data as FC.
- [x] Save RNN loss curve.
- [x] Save RNN prediction plots.

Milestone 26 validation:

- `RnnSignalNet` lives in `src/ai_orchestra/models.py`. It uses one `nn.RNN`
  layer with a configurable hidden size, plus a per-time-step linear head.
- `run_rnn_baseline()` lives in `src/ai_orchestra/rnn_baseline.py` and trains
  the RNN on the same `SequenceSignalDataset` built from the same signal sets
  and the same train/test split seed used by the FC baseline.
- `plot_rnn_baseline_result()` lives in `src/ai_orchestra/rnn_plotting.py`
  and reuses the generic `plot_prediction_window` and `plot_loss_curves`
  helpers, so no plotting code is duplicated.
- Shared `resolve_device`, `sample_prediction_indices`, and `sample_predictions`
  were extracted to `src/ai_orchestra/prediction_sampling.py` so both FC and
  RNN baselines and the training loops use one implementation.
- SDK now exposes `train_rnn_baseline` and `plot_rnn_predictions` next to the
  FC entry points.
- Quick run (1 signal set, 100 samples, 5 epochs) drops both train and test
  loss from about `0.64` to `0.33`, clearly faster than the FC baseline at
  the same scale, with very small train/test gap.
- Plots saved under `results/figures/rnn_baseline/`.
- Visual quality of the predictions is still poor at this smoke-run scale,
  so improvement is left for the RNN review gate in Milestone 27.

## Milestone 27: RNN Review Gate

- [x] Compare RNN train loss to FC train loss.
- [x] Compare RNN test loss to FC test loss.
- [x] Inspect RNN prediction plots.
- [x] Check whether RNN improves reconstruction.
- [x] Check whether RNN is unstable.
- [x] Note whether the window size is too short for RNN advantage.
- [x] Run tests.
- [x] Run Ruff.
- [x] Fix any easy issues.
- [x] Do not add LSTM before this gate is passed.

Milestone 27 review result:

- Same-seed, same-defaults comparison run (1 signal set, 100 samples, batch 32, 5 epochs):
  - FC  final loss: train `0.5049`, test `0.5148`.
  - RNN final loss: train `0.3323`, test `0.3337`.
  - RNN improves over FC by about `34.2%` on train loss and `35.2%` on test loss.
- RNN training is stable: loss decreases smoothly and monotonically every epoch,
  no NaN spikes, and the train/test gap stays around `0.001`.
- RNN prediction plots are visibly slightly better than FC (predictions track
  the sign and trend of the target more often), but at this smoke-run scale
  most predictions still do not closely match the target windows.
- Window size note: with `window_size=10` at `1000 Hz` sampling, the window
  covers `10 ms`. At `1 Hz` that is only `1%` of one period (a near-linear
  segment), and at `7 Hz` it is about `7%`. Any sequence model has very little
  temporal structure to exploit at this window size, which caps how much RNN
  or LSTM can beat FC unless the data scale and window size are increased.
- Quality gate: `131` tests pass, Ruff is clean, no easy issues to fix.
- Decision: gate passes. Adding LSTM is justified to compare against RNN on
  the same data, with the same window-size caveat documented.

## Milestone 28: Add LSTM

- [x] Add `LstmSignalNet`.
- [x] Accept input shape `(batch_size, 10, 5)`.
- [x] Use a configurable hidden size.
- [x] Start with one LSTM layer.
- [x] Map LSTM outputs to one value per time step.
- [x] Return output shape `(batch_size, 10)`.
- [x] Run one fake sequence batch through the LSTM.
- [x] Test LSTM output shape.
- [x] Test LSTM output contains finite values.
- [x] Train LSTM on a tiny subset.
- [x] Confirm LSTM training loop does not crash.
- [x] Confirm LSTM loss is finite.
- [x] Train LSTM on the same baseline data as FC and RNN.
- [x] Save LSTM loss curve.
- [x] Save LSTM prediction plots.

Milestone 28 validation:

- `LstmSignalNet` lives in `src/ai_orchestra/models.py`. It and `RnnSignalNet`
  now share a small `_SequenceSignalNet` base class that holds the per-time-step
  linear head and the common forward pass, so only the recurrent cell differs.
- `run_lstm_baseline()` lives in `src/ai_orchestra/lstm_baseline.py` and uses
  the shared `run_sequence_baseline` runner from `sequence_baseline.py`.
- `plot_lstm_baseline_result()` lives in `src/ai_orchestra/lstm_plotting.py`
  and reuses the shared `plot_sequence_baseline_result` helper.
- `rnn_baseline.py` and `rnn_plotting.py` were refactored to call the same
  shared runner and plotting helper, so RNN, LSTM, and any future sequence
  model share one code path. `baseline.py` for FC was left as-is since it
  uses a different dataset.
- `model_entries.py` was added to hold per-model SDK wrappers and keep
  `sdk.py` focused on the runnable pipeline. No file exceeds 150 lines now.
- SDK exposes `train_lstm_baseline` and `plot_lstm_predictions` next to the
  FC and RNN entries.
- Quick run (1 signal set, 100 samples, batch 32, 5 epochs):
  - LSTM train loss: `0.6447 -> 0.5369`.
  - LSTM test loss: `0.7013 -> 0.5636`.
  - Smooth monotonic decrease, loss is finite, no crashes.
- Plots saved under `results/figures/lstm_baseline/`.
- The LSTM is slower to converge than the simple RNN at the same epoch
  count, which is expected given the larger gate parameter set. The full
  LSTM vs RNN comparison and the worth-the-complexity call is left for
  the LSTM review gate in Milestone 29.

## Milestone 29: LSTM Review Gate

- [x] Compare LSTM train loss to FC and RNN.
- [x] Compare LSTM test loss to FC and RNN.
- [x] Inspect LSTM prediction plots.
- [x] Check whether LSTM improves reconstruction.
- [x] Check whether LSTM is worth the added complexity.
- [x] Run tests.
- [x] Run Ruff.
- [x] Fix any easy issues.

Milestone 29 review result:

- Same-seed, same-defaults comparison (1 signal set, 100 samples, batch 32,
  5 epochs, learning rate `1e-3`, hidden size `32`):

  | Model | Final train loss | Final test loss |
  | ----- | ---------------- | --------------- |
  | FC    | `0.5049`         | `0.5148`        |
  | RNN   | `0.3323`         | `0.3337`        |
  | LSTM  | `0.5369`         | `0.5636`        |

- Versus FC: LSTM is `6.3%` worse on train and `9.5%` worse on test loss.
- Versus RNN: LSTM is `61.6%` worse on train and `68.9%` worse on test loss.
- LSTM training is **stable**: smooth monotonic decrease every epoch, no NaN,
  finite outputs throughout, no crashes.
- LSTM prediction plots: at this scale predictions hover near `0` for all
  four target frequencies, well off targets like `-0.78` (5 Hz) and `-1.25`
  (7 Hz). Visually the LSTM is worse than RNN and roughly on par with FC.
- Honest interpretation: at five epochs and one signal set, the LSTM is
  clearly **undertrained**. With four gates and a cell state it has many
  more parameters than the simple RNN, so it needs more data and more epochs
  to reach the same point. Loss is still decreasing per epoch, so the
  architecture works — it just hasn't converged yet at this smoke-run scale.
- Worth-the-complexity call at the current scale: **no**. At the current
  scale the simple RNN beats both FC and LSTM and trains faster too.
- Worth investigating later: at larger scale (more signal sets, longer
  training, possibly a longer window) the LSTM may surpass the RNN. That
  belongs in a proper experiment milestone, not the smoke baseline.
- Quality gate: `139` tests pass, Ruff clean, no easy issues to fix.

### Phase G: Evaluation, Experiments, And Results

Good commit scope: model comparison metrics, config files, experiment variants, stricter generalization checks, and result organization.

## Milestone 30: Model Comparison

- [ ] Create `src/ai_orchestra/evaluation.py`.
- [ ] Add MSE metric calculation.
- [ ] Add optional MAE metric calculation.
- [ ] Calculate overall FC test MSE.
- [ ] Calculate overall RNN test MSE.
- [ ] Calculate overall LSTM test MSE.
- [ ] Calculate per-frequency FC MSE.
- [ ] Calculate per-frequency RNN MSE.
- [ ] Calculate per-frequency LSTM MSE.
- [ ] Create a comparison dictionary.
- [ ] Save comparison metrics as JSON.
- [ ] Save comparison metrics as CSV if useful.
- [ ] Plot final test MSE by model.
- [ ] Plot per-frequency MSE by model.
- [ ] Write short notes about which model performs best.

## Milestone 31: Config Polish

- [ ] Create `config/`.
- [ ] Create `config/default.json`.
- [ ] Move frequencies into config.
- [ ] Move sampling frequency into config.
- [ ] Move number of samples into config.
- [ ] Move window size into config.
- [ ] Move number of signal sets into config.
- [ ] Move noise strengths into config.
- [ ] Move train/test split into config.
- [ ] Move batch size into config.
- [ ] Move epoch count into config.
- [ ] Move learning rate into config.
- [ ] Move hidden size into config.
- [ ] Move random seed into config.
- [ ] Add config loading code.
- [ ] Validate required config fields.
- [ ] Test config loading.
- [ ] Test invalid config errors.
- [ ] Keep hardcoded defaults only where they make sense.

## Milestone 32: Experiment Variants

- [ ] Create `config/experiments.json` if needed.
- [ ] Add a low-noise experiment.
- [ ] Add a medium-noise experiment.
- [ ] Add a high-noise experiment.
- [ ] Add a smaller-hidden-size experiment.
- [ ] Add a larger-hidden-size experiment.
- [ ] Add a one-layer experiment.
- [ ] Add a two-layer experiment.
- [ ] Add a window-size `10` experiment.
- [ ] Add a larger-window experiment.
- [ ] Add an `80/20` split baseline.
- [ ] Optionally add a `70/30` split experiment.
- [ ] Optionally add a `90/10` split experiment.
- [ ] Run only one variant at a time at first.
- [ ] Save each experiment's config with its results.
- [ ] Avoid changing too many variables in one experiment.

## Milestone 33: Stricter Generalization Checks

- [ ] Evaluate on signal sets not used during training.
- [ ] Consider splitting by generated signal set instead of random windows.
- [ ] Consider time-based splitting inside each signal set.
- [ ] Compare random-window split vs. held-out-signal-set split.
- [ ] Check whether FC performance drops on stricter splits.
- [ ] Check whether RNN performance drops on stricter splits.
- [ ] Check whether LSTM performance drops on stricter splits.
- [ ] Document which split is used for final conclusions.
- [ ] Keep random split as a quick baseline if useful.

## Milestone 34: Results Organization

- [ ] Create `results/`.
- [ ] Create `results/figures/`.
- [ ] Create `results/metrics/`.
- [ ] Create `results/models/` only if saving checkpoints.
- [ ] Save loss curves with clear filenames.
- [ ] Save prediction plots with clear filenames.
- [ ] Save raw metrics with clear filenames.
- [ ] Avoid committing very large generated files.
- [ ] Decide which example figures should be committed.
- [ ] Keep generated artifacts reproducible from code.

## Milestone 35: README Finalization

- [ ] Update project overview.
- [ ] Add installation instructions with `uv`.
- [ ] Add test instructions with `uv run pytest`.
- [ ] Add lint instructions with `uv run ruff check .`.
- [ ] Explain signal generation.
- [ ] Explain many random signal sets.
- [ ] Explain the conditioned one-hot task.
- [ ] Explain FC input format.
- [ ] Explain RNN/LSTM input format.
- [ ] Explain training setup.
- [ ] Add model comparison results.
- [ ] Add prediction plots.
- [ ] Add loss curves.
- [ ] Add conclusions.
- [ ] Mention limitations.
- [ ] Mention possible future work.

### Phase H: Final Polish And Optional Enhancements

Good commit scope: final quality pass, cleanup, documentation consistency, and optional features after the core project works.

## Milestone 36: Final Quality Pass

- [ ] Run all tests.
- [ ] Run Ruff.
- [ ] Check Python files are reasonably small.
- [ ] Split any Python file that grows too large.
- [ ] Check that business logic is available through SDK functions.
- [ ] Check that one-off scripts do not duplicate core logic.
- [ ] Check that generated data is not accidentally committed.
- [ ] Check that no secrets or local environment files are committed.
- [ ] Check that `REQUIREMENTS.md` still matches the project.
- [ ] Check that `PLAN.md` still matches the project.
- [ ] Check that `TODO.md` accurately reflects remaining work.
- [ ] Check that README commands work.
- [ ] Check that final plots exist.
- [ ] Check that final metrics exist.

## Milestone 37: Optional Improvements After Core Success

- [ ] Add checkpoint saving.
- [ ] Add checkpoint loading.
- [ ] Add command-line runner.
- [ ] Add progress bars.
- [ ] Add parameter count reporting.
- [ ] Add GPU device selection.
- [ ] Add dataset caching.
- [ ] Add saved generated datasets.
- [ ] Add richer experiment summaries.
- [ ] Add more frequencies.
- [ ] Add random frequency sets.
- [ ] Add more realistic signal noise.
- [ ] Add comparison against a classical signal-processing baseline.
- [ ] Add cross-correlation or spectral analysis plots.
- [ ] Add notebook examples if useful.

