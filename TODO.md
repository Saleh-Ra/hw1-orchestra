# TODO

This checklist follows `REQUIREMENTS.md` and `PLAN.md`, with one important work-order correction:

Start with a minimal working ML pipeline first. Do not build all infrastructure before proving that the model can learn the reconstruction task.

## Status Legend

- `[ ]` Not started
- `[x]` Done
- Keep each item small enough to finish, test, or review independently.

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
- Full pytest/Ruff commands are still blocked until `uv` or local dev dependencies are available.

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
- Full pytest/Ruff commands are still blocked until `uv` or local dev dependencies are available.

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

- [ ] Add a small `SignalSet` dataclass.
- [ ] Store the time axis.
- [ ] Store the frequencies.
- [ ] Store the clean signals.
- [ ] Store the noisy signals.
- [ ] Store the clean mixed signal.
- [ ] Store the noisy mixed signal.
- [ ] Store amplitudes if useful for debugging.
- [ ] Store phases if useful for debugging.
- [ ] Add a function that returns one complete `SignalSet`.
- [ ] Test that `SignalSet` fields have expected shapes.
- [ ] Test that one complete signal set can be generated with one function call.

## Milestone 9: Generate Many Signal Sets

- [ ] Add a function to generate multiple random signal sets.
- [ ] Use `50` signal sets as the first default.
- [ ] Use a different seed stream per signal set.
- [ ] Keep frequencies the same across sets for the first baseline.
- [ ] Randomize amplitudes across sets.
- [ ] Randomize phases across sets.
- [ ] Randomize noise across sets.
- [ ] Return signal sets in a list or stacked structure.
- [ ] Test that the number of generated sets matches the request.
- [ ] Test that different sets are not identical.
- [ ] Test that generation is reproducible from the same base seed.
- [ ] Keep memory usage reasonable.
- [ ] Add a smaller test setting with only `2` or `3` signal sets.

## Milestone 10: Visualize Raw Signals Early

- [ ] Create `src/ai_orchestra/plotting.py`.
- [ ] Add a function to plot clean individual signals.
- [ ] Add a function to plot noisy individual signals.
- [ ] Add a function to plot clean mixed signal.
- [ ] Add a function to plot noisy mixed signal.
- [ ] Limit plotted samples so figures are readable.
- [ ] Save plots to a results folder.
- [ ] Ensure plotting works without an interactive display.
- [ ] Generate one raw clean-signal plot manually.
- [ ] Generate one noisy-signal plot manually.
- [ ] Generate one mixed-signal plot manually.
- [ ] Visually inspect the plots.
- [ ] Fix signal generation if plots look obviously wrong.

## Milestone 11: Build One-Hot Encoding

- [ ] Create `src/ai_orchestra/dataset.py`.
- [ ] Add a function to create one-hot vectors.
- [ ] Map class index `0` to `[1, 0, 0, 0]`.
- [ ] Map class index `1` to `[0, 1, 0, 0]`.
- [ ] Map class index `2` to `[0, 0, 1, 0]`.
- [ ] Map class index `3` to `[0, 0, 0, 1]`.
- [ ] Test every one-hot vector.
- [ ] Test invalid class indices fail clearly.
- [ ] Keep the one-hot length tied to the number of frequencies.

## Milestone 12: Build Window Indexing

- [ ] Add logic to compute valid window start positions.
- [ ] Use window size `10`.
- [ ] Confirm valid starts are `0` through `9990`.
- [ ] Confirm one signal set gives `9991` window positions.
- [ ] Add a function to extract one window.
- [ ] Test the first window.
- [ ] Test a middle window.
- [ ] Test the last valid window.
- [ ] Test that off-by-one errors are not present.
- [ ] Test invalid window starts fail clearly.

## Milestone 13: Build FC Dataset Items

- [ ] Create dataset logic for fully connected input format.
- [ ] For each signal set, iterate through valid window positions.
- [ ] For each window position, create one example per target frequency.
- [ ] Extract the noisy mixed signal window.
- [ ] Extract the matching clean target signal window.
- [ ] Append the one-hot vector to the noisy input window.
- [ ] Ensure FC input shape is `(14,)`.
- [ ] Ensure target shape is `(10,)`.
- [ ] Ensure values are returned as PyTorch tensors.
- [ ] Use float tensors.
- [ ] Test the first FC dataset item.
- [ ] Test an FC item for each condition vector.
- [ ] Test that `C = [1, 0, 0, 0]` targets `S1`.
- [ ] Test that `C = [0, 1, 0, 0]` targets `S2`.
- [ ] Test that `C = [0, 0, 1, 0]` targets `S3`.
- [ ] Test that `C = [0, 0, 0, 1]` targets `S4`.
- [ ] Test FC input length is exactly `14`.
- [ ] Test target length is exactly `10`.

## Milestone 14: Dataset Size Checks

- [ ] Compute examples per signal set.
- [ ] Confirm one set gives `9991 * 4 = 39964` examples.
- [ ] Confirm `50` sets gives `1998200` examples if fully materialized.
- [ ] Decide whether to materialize all examples or index lazily.
- [ ] Prefer lazy indexing if memory is too large.
- [ ] Implement lazy indexing if needed.
- [ ] Test dataset length with one set.
- [ ] Test dataset length with two sets.
- [ ] Test dataset length with fifty sets or a simulated equivalent.
- [ ] Make sure length calculation is fast.
- [ ] Make sure item retrieval is fast enough for training.

## Milestone 15: Minimal Train/Test Split

- [ ] Add deterministic train/test split.
- [ ] Use `80/20` as the default.
- [ ] Use a random seed for the split.
- [ ] Start with random split for the minimal pipeline.
- [ ] Document that time-based split may be added later.
- [ ] Confirm train set length is about 80%.
- [ ] Confirm test set length is about 20%.
- [ ] Confirm no index appears in both sets.
- [ ] Test split reproducibility.
- [ ] Test split ratio can be changed later.

## Milestone 16: Create DataLoader

- [ ] Add PyTorch `DataLoader` creation.
- [ ] Use shuffle for the training loader.
- [ ] Do not shuffle the test loader.
- [ ] Set a simple default batch size.
- [ ] Fetch one training batch.
- [ ] Confirm batch input shape is `(batch_size, 14)`.
- [ ] Confirm batch target shape is `(batch_size, 10)`.
- [ ] Confirm batch tensors are finite.
- [ ] Add a small test for DataLoader output shapes.

## Milestone 17: Build The First FC Model

- [ ] Create `src/ai_orchestra/models.py`.
- [ ] Add `FullyConnectedSignalNet`.
- [ ] Set input size to `14`.
- [ ] Set output size to `10`.
- [ ] Add one hidden layer first.
- [ ] Use ReLU activation.
- [ ] Keep the model simple for the first run.
- [ ] Run one fake batch through the model.
- [ ] Confirm output shape is `(batch_size, 10)`.
- [ ] Test model forward pass.
- [ ] Test model output contains finite values.

## Milestone 18: Minimal Training Loop

- [ ] Create `src/ai_orchestra/training.py`.
- [ ] Add a function to train one epoch.
- [ ] Add a function to evaluate one epoch.
- [ ] Use MSE loss.
- [ ] Use Adam optimizer.
- [ ] Move model to CPU by default.
- [ ] Support GPU if available without requiring it.
- [ ] Track average train loss.
- [ ] Track average test loss.
- [ ] Print or return epoch metrics.
- [ ] Train the FC model for one epoch on a tiny subset.
- [ ] Confirm the training loop does not crash.
- [ ] Confirm loss is finite.
- [ ] Add an integration smoke test for one tiny FC training run.

## Milestone 19: First Overfit Test

- [ ] Create a tiny dataset from one or two generated signal sets.
- [ ] Train the FC model on a very small subset.
- [ ] Use enough epochs to see if the model can overfit.
- [ ] Track training loss.
- [ ] Confirm training loss decreases.
- [ ] Plot one prediction from the overfit run.
- [ ] Compare predicted window to target window.
- [ ] If it cannot overfit, inspect dataset indexing.
- [ ] If it cannot overfit, inspect target mapping.
- [ ] If it cannot overfit, inspect model output shape.
- [ ] If it cannot overfit, inspect loss calculation.
- [ ] Do not move to larger training until this works.

## Milestone 20: First Real FC Baseline

- [ ] Generate many signal sets for training.
- [ ] Start with fewer than `50` sets if runtime is too high.
- [ ] Increase toward `50` sets after the pipeline works.
- [ ] Build the FC dataset.
- [ ] Build train and test splits.
- [ ] Train the FC model for a small number of epochs.
- [ ] Record train loss per epoch.
- [ ] Record test loss per epoch.
- [ ] Confirm train loss decreases.
- [ ] Confirm test loss is finite.
- [ ] Save the loss history in memory first.
- [ ] Save metrics to disk only after the run works.
- [ ] Inspect several predictions.
- [ ] Confirm predictions roughly follow the target signal.
- [ ] Note failure modes if reconstruction is poor.

## Milestone 21: FC Prediction Plots

- [ ] Add a function to plot target vs. prediction.
- [ ] Plot a prediction for `S1`.
- [ ] Plot a prediction for `S2`.
- [ ] Plot a prediction for `S3`.
- [ ] Plot a prediction for `S4`.
- [ ] Include the model name in the plot title.
- [ ] Include the requested frequency in the plot title.
- [ ] Save plots under `results/figures/`.
- [ ] Visually inspect all four FC prediction plots.
- [ ] Add a function to plot training and test loss curves.
- [ ] Save the FC loss curve.
- [ ] Use these plots to decide whether the FC baseline is good enough.

## Milestone 22: Minimal SDK Or Runner

- [ ] Create a simple runnable entry point.
- [ ] Prefer a small SDK function before adding a full CLI.
- [ ] Create `src/ai_orchestra/sdk.py`.
- [ ] Add a function to run signal generation.
- [ ] Add a function to build datasets.
- [ ] Add a function to train the FC baseline.
- [ ] Add a function to plot FC predictions.
- [ ] Create a simple script only if needed for convenience.
- [ ] Ensure the runner calls package functions instead of duplicating logic.
- [ ] Run the full minimal FC pipeline from one place.
- [ ] Confirm the run produces metrics and plots.

## Milestone 23: Minimal README Update

- [ ] Create or update `README.md`.
- [ ] Explain the project in one paragraph.
- [ ] Explain the minimal pipeline.
- [ ] Explain how signals are generated.
- [ ] Explain how the one-hot condition works.
- [ ] Explain how to run tests.
- [ ] Explain how to run the FC baseline.
- [ ] Add the current FC result after it exists.
- [ ] Add at least one prediction plot after it exists.
- [ ] Keep README honest about what is implemented so far.

## Milestone 24: FC Baseline Review Gate

- [ ] Confirm signal generation works.
- [ ] Confirm many signal sets are generated.
- [ ] Confirm dataset indexing works.
- [ ] Confirm FC model trains without crashing.
- [ ] Confirm FC training loss decreases.
- [ ] Confirm FC test loss is finite.
- [ ] Confirm FC prediction plots are generated.
- [ ] Confirm at least some FC predictions visually resemble targets.
- [ ] Run all tests.
- [ ] Run Ruff.
- [ ] Fix any easy issues.
- [ ] Decide whether to improve FC before adding RNN.
- [ ] Do not add RNN before this gate is passed.

## Milestone 25: Prepare Sequence Dataset Format

- [ ] Add sequence-format dataset support.
- [ ] Keep the same underlying signal sets.
- [ ] Keep the same target windows.
- [ ] For each time step, include the noisy sample.
- [ ] For each time step, include the full one-hot vector.
- [ ] Ensure each sequence input shape is `(10, 5)`.
- [ ] Ensure each sequence target shape is `(10,)`.
- [ ] Test sequence shape for one item.
- [ ] Test sequence shape for one batch.
- [ ] Test sequence target mapping for each class.
- [ ] Reuse as much indexing logic as possible.
- [ ] Avoid duplicating target-mapping logic.

## Milestone 26: Add Simple RNN

- [ ] Add `RnnSignalNet`.
- [ ] Accept input shape `(batch_size, 10, 5)`.
- [ ] Use a configurable hidden size.
- [ ] Start with one recurrent layer.
- [ ] Map recurrent outputs to one value per time step.
- [ ] Return output shape `(batch_size, 10)`.
- [ ] Run one fake sequence batch through the RNN.
- [ ] Test RNN output shape.
- [ ] Test RNN output contains finite values.
- [ ] Train RNN on a tiny subset.
- [ ] Confirm RNN training loop does not crash.
- [ ] Confirm RNN loss is finite.
- [ ] Train RNN on the same baseline data as FC.
- [ ] Save RNN loss curve.
- [ ] Save RNN prediction plots.

## Milestone 27: RNN Review Gate

- [ ] Compare RNN train loss to FC train loss.
- [ ] Compare RNN test loss to FC test loss.
- [ ] Inspect RNN prediction plots.
- [ ] Check whether RNN improves reconstruction.
- [ ] Check whether RNN is unstable.
- [ ] Note whether the window size is too short for RNN advantage.
- [ ] Run tests.
- [ ] Run Ruff.
- [ ] Fix any easy issues.
- [ ] Do not add LSTM before this gate is passed.

## Milestone 28: Add LSTM

- [ ] Add `LstmSignalNet`.
- [ ] Accept input shape `(batch_size, 10, 5)`.
- [ ] Use a configurable hidden size.
- [ ] Start with one LSTM layer.
- [ ] Map LSTM outputs to one value per time step.
- [ ] Return output shape `(batch_size, 10)`.
- [ ] Run one fake sequence batch through the LSTM.
- [ ] Test LSTM output shape.
- [ ] Test LSTM output contains finite values.
- [ ] Train LSTM on a tiny subset.
- [ ] Confirm LSTM training loop does not crash.
- [ ] Confirm LSTM loss is finite.
- [ ] Train LSTM on the same baseline data as FC and RNN.
- [ ] Save LSTM loss curve.
- [ ] Save LSTM prediction plots.

## Milestone 29: LSTM Review Gate

- [ ] Compare LSTM train loss to FC and RNN.
- [ ] Compare LSTM test loss to FC and RNN.
- [ ] Inspect LSTM prediction plots.
- [ ] Check whether LSTM improves reconstruction.
- [ ] Check whether LSTM is worth the added complexity.
- [ ] Run tests.
- [ ] Run Ruff.
- [ ] Fix any easy issues.

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

