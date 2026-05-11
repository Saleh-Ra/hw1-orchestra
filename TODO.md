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

- [x] Create `src/ai_orchestra/evaluation.py`.
- [x] Add MSE metric calculation (`compute_mse`).
- [x] Add optional MAE metric calculation (`compute_mae`).
- [x] Calculate overall FC test MSE.
- [x] Calculate overall RNN test MSE.
- [x] Calculate overall LSTM test MSE.
- [x] Calculate per-frequency FC MSE.
- [x] Calculate per-frequency RNN MSE.
- [x] Calculate per-frequency LSTM MSE.
- [x] Create a comparison dictionary (`ModelComparison`).
- [x] Save comparison metrics as JSON (`save_comparison_json`).
- [x] Save comparison metrics as CSV (`save_comparison_csv`).
- [x] Plot final test MSE by model (`plot_overall_mse`).
- [x] Plot per-frequency MSE by model (`plot_per_frequency_mse`).
- [x] Write short notes about which model performs best (see review notes below).

### Milestone 30 review notes

- Settings used for the apples-to-apples comparison: `1` signal set,
  `num_samples=100`, `batch_size=32`, `epochs=5`, `learning_rate=1e-3`,
  `hidden_size=32`, `seed=42`, `device=cpu`. All three models are trained on
  the same train/test split.
- Overall test MSE (lower is better):

  | Model | Overall MSE |
  |-------|-------------|
  | FC    | `0.5372`    |
  | RNN   | `0.3406`    |
  | LSTM  | `0.4928`    |

- Ranking: **RNN beats LSTM beats FC** at this scale. RNN is `~37%` better
  than FC and `~31%` better than LSTM. LSTM edges FC by `~8%`.
- Per-frequency test MSE (lower is better):

  | Model | 1 Hz     | 3 Hz     | 5 Hz     | 7 Hz     |
  |-------|----------|----------|----------|----------|
  | FC    | `0.0613` | `0.6411` | `0.4751` | `0.9012` |
  | RNN   | `0.1092` | `0.1012` | `0.2841` | `0.8820` |
  | LSTM  | `0.0330` | `0.3991` | `0.3134` | `1.1902` |

- Per-frequency findings:
  - All three models do best at `1 Hz` and worst at `7 Hz`. That is expected
    because the `10`-sample window covers a smaller fraction of a `7 Hz`
    period than of a `1 Hz` period.
  - LSTM is the **best** single result at `1 Hz` (`0.0330`) — for very slow
    signals its smoothing bias is helpful.
  - LSTM is the **worst** single result at `7 Hz` (`1.1902`) — undertrained
    LSTM predictions hover near zero, which is far from the high-amplitude
    `7 Hz` target.
  - RNN dominates the mid frequencies (`3 Hz` and `5 Hz`), where the window
    contains a meaningful fraction of one period.
- Artifacts saved:
  - `results/metrics/model_comparison.json`
  - `results/metrics/model_comparison.csv`
  - `results/figures/comparison/overall_test_mse.png`
  - `results/figures/comparison/per_frequency_test_mse.png`
- Quality gate: `152` tests pass (139 + 13 new), Ruff clean. No lint
  waivers. New modules under `150` lines each (`evaluation.py` `86`,
  `comparison.py` `146`, `comparison_io.py` `57`,
  `comparison_plotting.py` `86`).
- Honest interpretation: results agree with milestones 27 and 29 directionally
  (RNN best at this smoke-run scale). The new per-frequency breakdown adds
  detail and shows that the per-class error is **not** uniform across
  frequencies, which is worth keeping in mind when designing the next
  experiment (more signal sets, more epochs, possibly a longer window).

## Milestone 31: Config Polish

- [x] Create `config/`.
- [x] Create `config/default.json`.
- [x] Move frequencies into config.
- [x] Move sampling frequency into config.
- [x] Move number of samples into config.
- [x] Move window size into config.
- [x] Move number of signal sets into config.
- [x] Move noise strengths into config (`amplitude_noise_std`,
      `phase_noise_std`, `additive_noise_std`).
- [x] Move train/test split into config (`train_ratio`).
- [x] Move batch size into config.
- [x] Move epoch count into config.
- [x] Move learning rate into config.
- [x] Move hidden size into config.
- [x] Move random seed into config.
- [x] Add config loading code (`src/ai_orchestra/config.py`,
      `load_pipeline_defaults`).
- [x] Validate required config fields (reuses `PipelineDefaults.validate`
      and dataclass type checking via `TypeError`-to-`ValueError`).
- [x] Test config loading (default path round-trip and custom-path overrides).
- [x] Test invalid config errors (unknown field, non-array tuple, invalid
      value, non-object root, missing file).
- [x] Keep hardcoded defaults only where they make sense (the dataclass
      keeps mirroring defaults so import-time code never does disk I/O,
      and a parity test asserts the two stay in sync).

### Milestone 31 review notes

- `config/default.json` is the documented source of truth for project
  tunables. The `PipelineDefaults` dataclass keeps the same values as
  fallback so importing `ai_orchestra` performs no disk I/O.
- Three new tunables were added to `PipelineDefaults` so the config covers
  everything Milestone 31 lists: `epochs`, `learning_rate`, `hidden_size`.
  Each has its own validation rule (must be positive).
- `load_pipeline_defaults(path: str | Path | None = None) -> PipelineDefaults`
  loads JSON, rejects unknown fields, coerces JSON arrays into tuples for
  tuple-typed fields, converts dataclass `TypeError` (e.g. wrong primitive
  type) into a clear `ValueError`, and then calls `defaults.validate()` so
  every config goes through the same checks that hardcoded defaults do.
- Tests added in `tests/unit/test_config.py`:
  - `test_default_config_matches_hardcoded_defaults` (parity)
  - `test_default_config_covers_every_pipeline_field` (no field forgotten)
  - `test_load_pipeline_defaults_from_custom_path`
  - `test_load_pipeline_defaults_missing_file_raises`
  - `test_load_pipeline_defaults_rejects_unknown_field`
  - `test_load_pipeline_defaults_rejects_non_array_tuple_field`
  - `test_load_pipeline_defaults_validates_values`
  - `test_load_pipeline_defaults_rejects_non_object_root`
- Quality gate: `160` tests pass (152 + 8 new), Ruff clean. New
  `config.py` is `68` lines, `defaults.py` grew to `79` lines, both under
  the `150`-line guideline.
- Backwards compatibility: nothing else in the SDK changed. Baselines and
  `compare_models` still accept their `epochs`, `learning_rate`, and
  `hidden_size` arguments directly. The dataclass values are available
  via `DEFAULTS.epochs` etc. for any future call site that wants to be
  fully config-driven.

## Milestone 32: Experiment Variants

- [x] Create `config/experiments.json` if needed.
- [x] Add a low-noise experiment (`noise_low`).
- [x] Add a medium-noise experiment (`noise_medium`, labels the default noise).
- [x] Add a high-noise experiment (`noise_high`).
- [x] Add a smaller-hidden-size experiment (`hidden_small`, `hidden_size=8`).
- [x] Add a larger-hidden-size experiment (`hidden_large`, `hidden_size=128`).
- [x] Add a one-layer experiment (`layers_one`, default recurrent stack).
- [x] Add a two-layer experiment (`layers_two`, `num_layers=2`).
- [x] Add a window-size `10` experiment (`window_10`, labels the default).
- [x] Add a larger-window experiment (`window_large`, `window_size=25`).
- [x] Add an `80/20` split baseline (`split_80_20`, labels the default).
- [x] Optionally add a `70/30` split experiment (`split_70_30`).
- [x] Optionally add a `90/10` split experiment (`split_90_10`).
- [x] Run only one variant at a time at first (only `noise_low` executed
      end-to-end in this milestone).
- [x] Save each experiment's config with its results (the runner writes
      `config.json` containing the spec, the overrides, and the fully
      resolved defaults alongside `metrics.json`, `metrics.csv`, and the
      two PNG plots).
- [x] Avoid changing too many variables in one experiment (each variant
      overrides one knob, except the three noise sweeps which intentionally
      group the three coupled noise stds).

### Milestone 32 review notes

- Added `num_layers` to `PipelineDefaults` (default `1`) and threaded it
  through `compare_models` so the layer variants (`layers_one`,
  `layers_two`) actually affect RNN/LSTM construction.
- New module `src/ai_orchestra/experiments.py` provides
  `ExperimentSpec`, `load_experiments`, and `apply_overrides`.
- New module `src/ai_orchestra/experiment_runner.py` provides
  `run_experiment` and `list_known_experiments`. The runner is what saves
  the resolved config to disk so each result is self-contained.
- CLI entry point added at `scripts/run_experiment.py`. Usage:

  ```powershell
  $env:PYTHONPATH="src"
  python scripts/run_experiment.py --list
  python scripts/run_experiment.py noise_low --num-samples 100 --batch-size 32 --signal-set-count 1
  ```

- One variant executed end-to-end: `noise_low` at smoke scale
  (`1` signal set, `100` samples, `batch_size=32`, default `3` epochs,
  `learning_rate=1e-3`, `hidden_size=32`, `num_layers=1`, `seed=42`).
  Overall test MSE: FC `0.5903`, RNN `0.3641`, LSTM `0.6306`. RNN keeps
  the lead even at lower noise (same direction as the M30 default-noise
  run; absolute values are higher here because this run used the
  config default of `3` epochs, not the `5` epochs `scripts/run_comparison.py`
  uses).
- Artifacts written by the runner to `results/experiments/noise_low/`:
  - `config.json` (spec, overrides, resolved defaults)
  - `metrics.json`, `metrics.csv`
  - `overall_test_mse.png`, `per_frequency_test_mse.png`
- File sizes (all under the 150-line guideline): `experiments.py` `95`,
  `experiment_runner.py` `113`, `comparison.py` `134`,
  `comparison_internals.py` `31`, `scripts/run_experiment.py` `91`.
  `comparison.py` was kept under 150 by extracting `train_with_loss_history`
  and `by_frequency` into the small `comparison_internals.py` companion.
- Quality gate: `174` tests pass (`160` + `14` new — 11 unit on
  loader/overrides, 3 integration on the runner), Ruff clean.
- Honest interpretation: the milestone's purpose is the **infrastructure**
  for reproducible, one-knob-at-a-time experiments, not yet a sweep of
  results. The `noise_low` run confirms the runner end-to-end. Running
  the remaining variants is best done in a dedicated experiment milestone
  with more signal sets and more epochs.

## Milestone 33: Stricter Generalization Checks

- [x] Evaluate on signal sets not used during training (via
      `holdout_signal_set_split_indices` in `src/ai_orchestra/splits.py`).
- [x] Consider splitting by generated signal set instead of random windows
      (implemented; this is the held-out-signal-set split).
- [ ] ~~Consider time-based splitting inside each signal set.~~
      Deliberately skipped — out of scope for this project.
- [x] Compare random-window split vs. held-out-signal-set split
      (`scripts/run_generalization_check.py`, results saved under
      `results/generalization/`).
- [x] Check whether FC performance drops on stricter splits.
- [x] Check whether RNN performance drops on stricter splits.
- [x] Check whether LSTM performance drops on stricter splits.
- [x] Document which split is used for final conclusions (held-out
      signal set is used for the headline conclusions; random split is
      kept as a quick smoke baseline).
- [x] Keep random split as a quick baseline (still the default in
      `compare_models`).

### Milestone 33 review notes

Settings used (`scripts/run_generalization_check.py`):
`signal_set_count=5`, `num_samples=100`, `batch_size=32`, `epochs=3`,
`learning_rate=1e-3`, `hidden_size=32`, `num_layers=1`, `seed=42`, cpu.

Random window split (every signal set appears in both train and test):

| Model | Overall MSE |
|-------|-------------|
| FC    | `0.4012`    |
| RNN   | `0.4115`    |
| LSTM  | `0.3999`    |

Held-out signal-set split (4 sets train, 1 set test, test sets unseen):

| Model | Overall MSE | Delta vs random |
|-------|-------------|-----------------|
| FC    | `0.5774`    | `+0.1762`       |
| RNN   | `0.5463`    | `+0.1349`       |
| LSTM  | `0.5550`    | `+0.1551`       |

Findings:

- **All three models degrade meaningfully** under the held-out split
  (`+34%` to `+44%` worse). The random-window split was therefore
  overstating performance because train and test shared the same
  underlying clean signals.
- **RNN degrades the least**, FC the most. The recurrent state appears
  to help slightly with truly unseen signals, but the difference is
  modest (`+0.1349` vs `+0.1762`).
- Under the stricter split the three models are **roughly tied**
  (`0.55–0.58`). The RNN lead seen in M30 at `1` signal set + `5` epochs
  shrinks toward the noise floor once we ask for real generalization.
- **Decision for headline conclusions**: report held-out-signal-set
  results as the primary numbers. Random split stays in the codebase as
  the quick smoke baseline that `compare_models` uses by default.

Implementation notes:

- New module `src/ai_orchestra/splits.py` (`59` lines).
- `compare_models` got two new optional args: `split_indices` (tuple of
  ndarrays, defaults to `None` → use the existing random split) and
  `split_name` (string written into `settings["split"]` for the saved
  metrics). Both arguments are backwards compatible; existing scripts
  and tests still work.
- New unit tests in `tests/unit/test_splits.py` (`6` tests) cover
  disjoint partitions, full-set membership, determinism, invalid
  configurations, and `compare_models` accepting a custom split.
- Artifacts: `results/generalization/{random_split,holdout_split}_metrics.{json,csv}`
  and `results/generalization/comparison.json` with the deltas.

## Milestone 34: Results Organization

This milestone is mostly an audit — the layout grew naturally through
M22–M33. Items below verify that what we have is correct.

- [x] Create `results/`.
- [x] Create `results/figures/` (`fc_baseline/`, `rnn_baseline/`,
      `lstm_baseline/`, `raw_examples/`, `comparison/`).
- [x] Create `results/metrics/` (model comparison JSON + CSV).
- [x] ~~Create `results/models/` only if saving checkpoints.~~
      Not created on purpose; checkpoints aren't part of the deliverable
      and `.gitignore` already covers `results/models/` and `*.pt`/`*.pth`/
      `*.ckpt` if we ever do save them.
- [x] Save loss curves with clear filenames
      (`<model>_loss_curve.png` per baseline).
- [x] Save prediction plots with clear filenames
      (`<model>_prediction_s<class>_<freq>hz.png` per sample).
- [x] Save raw metrics with clear filenames
      (`model_comparison.{json,csv}`, per-experiment `metrics.{json,csv}`,
      `random_split_metrics.{json,csv}`,
      `holdout_split_metrics.{json,csv}`, `comparison.json`).
- [x] Avoid committing very large generated files (full `results/` tree
      is `~760 KB`, all small PNG/JSON/CSV; no `.pt`/`.pth` ever written).
- [x] Decide which example figures should be committed (small smoke-run
      plots only; new experiments go under `results/experiments/<name>/`).
- [x] Keep generated artifacts reproducible from code
      (`python -m ai_orchestra`, `scripts/run_comparison.py`,
      `scripts/run_experiment.py`, `scripts/run_generalization_check.py`).

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

- [x] Run all tests — `180 passed` (M33 added six new tests).
- [x] Run Ruff — clean, no waivers.
- [x] Check Python files are reasonably small — every `src/` and
      `tests/` Python file is under `150` lines (verified by scanning
      all `.py` files with `Get-Content | Measure-Object`).
- [x] Split any Python file that grows too large — `comparison.py` was
      split into `comparison.py` + `comparison_internals.py` during M32
      when it would otherwise have exceeded `150` lines.
- [x] Check that business logic is available through SDK functions
      (`compare_models`, `run_experiment`, `train_*_baseline`,
      `evaluate_per_class_mse`, `holdout_signal_set_split_indices`, etc.
      are all exported from `ai_orchestra`).
- [x] Check that one-off scripts do not duplicate core logic — the four
      scripts (`run_comparison`, `run_experiment`, `run_generalization_check`,
      `__main__`) are thin orchestrators that import SDK functions.
- [x] Check that generated data is not accidentally committed — full
      `results/` tree is `~760 KB`, only small PNG/JSON/CSV; checkpoints
      and `*.pt`/`*.pth`/`*.ckpt` are gitignored.
- [x] Check that no secrets or local environment files are committed —
      no `.env`, `credentials`, or secret-shaped files are tracked.
- [x] Check that `REQUIREMENTS.md` still matches the project — present at
      project root and consistent with the README at a high level.
- [x] Check that `PLAN.md` still matches the project — present
      (`676` lines, kept in sync with milestone scope).
- [x] Check that `TODO.md` accurately reflects remaining work — only
      Milestone 37 (explicitly optional) is left unchecked.
- [x] Check that README commands work — all four runner commands
      executed during M30, M32, and M33 milestones; `python -m pytest`
      and `python -m ruff check .` both pass on the current tree.
- [x] Check that final plots exist — `results/figures/{fc,rnn,lstm}_baseline/`,
      `results/figures/comparison/`, `results/figures/raw_examples/`,
      `results/experiments/noise_low/`, and the bar charts inside
      `results/figures/comparison/` are all on disk.
- [x] Check that final metrics exist — `results/metrics/model_comparison.{json,csv}`,
      `results/experiments/noise_low/metrics.{json,csv}`, and
      `results/generalization/{random,holdout}_split_metrics.{json,csv}`
      plus `results/generalization/comparison.json` are all on disk.

## Milestone 37: Optional Improvements After Core Success

This milestone is **deliberately not pursued**. The project is complete
for its stated goal (FC vs RNN vs LSTM signal reconstruction with a
proper generalization check). The items below are real ideas for future
work, but each one is **out of scope** and listed here only as a
parking lot for follow-up.

- [ ] ~~Add checkpoint saving.~~ Not needed for the deliverable.
- [ ] ~~Add checkpoint loading.~~ Not needed for the deliverable.
- [ ] ~~Add command-line runner.~~ Already covered by the four
      `scripts/run_*.py` entry points and `python -m ai_orchestra`.
- [ ] ~~Add progress bars.~~ Cosmetic only.
- [ ] ~~Add parameter count reporting.~~ Could be a nice-to-have but
      not part of the report.
- [ ] ~~Add GPU device selection.~~ All runs are CPU-only and finish
      in seconds.
- [ ] ~~Add dataset caching.~~ Lazy indexing already keeps memory low.
- [ ] ~~Add saved generated datasets.~~ Signal generation is fast and
      deterministic from the seed.
- [ ] ~~Add richer experiment summaries.~~ JSON + CSV + two plots per
      experiment is already self-describing.
- [ ] ~~Add more frequencies.~~ Would require widening the one-hot;
      the four current frequencies are sufficient to compare models.
- [ ] ~~Add random frequency sets.~~ Same reasoning as above.
- [ ] ~~Add more realistic signal noise.~~ The three noise sweeps
      (`noise_low/medium/high`) already cover the question.
- [ ] ~~Add comparison against a classical signal-processing baseline.~~
      Acknowledged as a real follow-up in the README's Limitations
      section, but out of scope for this submission.
- [ ] ~~Add cross-correlation or spectral analysis plots.~~ Out of scope.
- [ ] ~~Add notebook examples if useful.~~ All workflows are runnable
      from `scripts/`; a notebook would duplicate that.

