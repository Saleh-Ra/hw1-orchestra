# Project Plan

## 1. Purpose

Build a clean machine learning project that trains neural networks to recover a requested clean sine wave from a noisy mixed signal.

The project will generate synthetic sine signals, combine them, add noise, create sliding-window training samples, train multiple model architectures, and compare their ability to reconstruct the requested clean signal.

The first target frequencies are `1 Hz`, `3 Hz`, `5 Hz`, and `7 Hz`. These are not special frequencies; they are the initial four signal classes chosen for the experiment. The implementation should make it easy to replace or extend them later.

## 2. Main Decisions

For the first implementation, use these defaults:

- Framework: PyTorch
- Package/dependency workflow: `uv`
- Default train/test split: `80/20`
- Default sampling frequency: `1000 Hz`
- Default signal length: `10000` samples
- Default duration: `10 seconds`
- Default window size: `10` samples
- Default frequencies: `[1, 3, 5, 7]`
- Default loss function: Mean Squared Error
- Default model comparison: fully connected network vs. simple RNN vs. LSTM
- Default dataset strategy: generate synthetic data programmatically, with optional saving later if useful
- Default experiment control: use config files so we can vary noise, model size, layers, window size, and split ratio without rewriting code

These decisions are allowed to change later if results show a better direction.

## 3. Expected Learning Task

Each training example should teach the model this behavior:

Given:

- A noisy window from the mixed signal.
- A one-hot vector identifying the desired frequency.

Predict:

- The clean signal window for that selected frequency at the same time position.

Example:

- Noisy mixed window: samples `6203` to `6212`
- Condition vector: `[0, 1, 0, 0]`
- Requested signal: `S2`, the `3 Hz` signal
- Target: clean `S2` samples `6203` to `6212`

The project is therefore a conditioned signal-reconstruction problem.

## 4. Repository Structure

Target structure:

```text
.
├── README.md
├── REQUIREMENTS.md
├── PLAN.md
├── pyproject.toml
├── uv.lock
├── config/
│   ├── default.json
│   └── experiments.json
├── src/
│   └── ai_orchestra/
│       ├── __init__.py
│       ├── sdk.py
│       ├── config.py
│       ├── signals.py
│       ├── dataset.py
│       ├── models.py
│       ├── training.py
│       ├── evaluation.py
│       ├── plotting.py
│       └── experiments.py
├── tests/
│   ├── unit/
│   │   ├── test_signals.py
│   │   ├── test_dataset.py
│   │   └── test_models.py
│   └── integration/
│       └── test_training_smoke.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
└── data/
    └── generated/
```

Notes:

- `src/ai_orchestra/sdk.py` should be the main entry point for business logic.
- CLI scripts, notebooks, or one-off runners should call the SDK instead of duplicating logic.
- Keep modules small and focused.
- Keep Python files under 150 lines where practical by splitting responsibilities.
- Store experiment outputs under `results/`.
- Store generated datasets under `data/generated/` only if we decide persistence is useful.

## 5. Configuration Plan

Create `config/default.json` for stable defaults:

- Frequencies
- Sampling frequency
- Number of samples
- Window size
- Noise settings
- Train/test split
- Batch size
- Epoch count
- Learning rate
- Hidden size
- Number of recurrent layers
- Random seed
- Output directories

Create `config/experiments.json` for experiment variants:

- Different noise levels
- Different hidden sizes
- Different numbers of layers
- Different window sizes
- Optional train/test split variants such as `70/30`, `80/20`, and `90/10`

The first working run should use the default config only. Experiment variants can come after the baseline pipeline works.

## 6. Implementation Phases

### Phase 1: Project Setup

Goals:

- Create the Python package layout.
- Add project metadata.
- Add dependencies.
- Add basic tests and lint configuration.

Tasks:

- Create `pyproject.toml`.
- Add dependencies with `uv`.
- Add PyTorch, NumPy, Matplotlib, and pytest.
- Add Ruff configuration.
- Create the `src/ai_orchestra/` package.
- Create `tests/unit/` and `tests/integration/`.
- Add a minimal `README.md`.
- Add `.gitignore` entries for generated results, model checkpoints, caches, and local environment files.

Acceptance criteria:

- `uv run pytest` runs successfully.
- `uv run ruff check .` runs successfully.
- The package can be imported.

### Phase 2: Signal Generation

Goals:

- Generate clean sine signals.
- Generate noisy versions of those signals.
- Generate clean and noisy mixed signals.

Implementation file:

- `src/ai_orchestra/signals.py`

Core functions:

- `make_time_axis(config) -> np.ndarray`
- `generate_clean_signal(frequency, time, amplitude, phase) -> np.ndarray`
- `generate_signal_set(config, seed) -> SignalSet`
- `combine_signals(signals) -> np.ndarray`

Data model:

- Use a small dataclass such as `SignalSet`.
- Store frequencies, time axis, clean signals, noisy signals, clean mix, and noisy mix.

Noise behavior:

- Use random amplitude per signal.
- Use random phase per signal.
- Add amplitude noise and phase noise to create noisy signal versions.
- Keep `beta` or phase values in the range `0` to `2pi`.
- Make the exact formula simple and understandable.

Acceptance criteria:

- Four clean signals are generated with shape `(4, 10000)`.
- Four noisy signals are generated with shape `(4, 10000)`.
- Clean and noisy mixed signals are generated with shape `(10000,)`.
- Generation is reproducible when the same seed is used.
- Unit tests verify shapes, reproducibility, and frequency count.

### Phase 3: Dataset Preparation

Goals:

- Convert generated signals into supervised sliding-window examples.
- Support both fully connected and sequence model input formats.

Implementation file:

- `src/ai_orchestra/dataset.py`

Dataset behavior:

- For each valid window position, create examples for all four target signals.
- With `10000` samples and a window size of `10`, there are `9991` possible window positions if the last start index is included.
- With four condition vectors, one generated signal set can produce `9991 * 4 = 39964` training examples.

Target behavior:

- The noisy mixed window is always the input signal source.
- The condition vector chooses which clean signal becomes the target.
- The target window always comes from the same time position as the input window.

Input formats:

- Fully connected input shape per item: `(14,)`
- Fully connected target shape per item: `(10,)`
- Sequence input shape per item: `(10, 5)`
- Sequence target shape per item: `(10,)`

Implementation options:

- Create one dataset class that can return either `fc` or `sequence` mode.
- Or create separate lightweight dataset wrappers for each mode if that keeps the code clearer.

Acceptance criteria:

- Dataset length is correct.
- One-hot vectors map to the correct target signal.
- Window slicing has no off-by-one error.
- Unit tests verify at least one known sample manually.
- Unit tests verify FC and sequence shapes.

### Phase 4: Train/Test Split

Goals:

- Split generated examples into train and test sets.
- Keep an 80/20 split as the baseline.
- Make the split ratio configurable.

Implementation location:

- `src/ai_orchestra/dataset.py` or `src/ai_orchestra/training.py`

Important detail:

- Avoid data leakage where almost identical neighboring windows appear in both train and test sets if this makes evaluation too optimistic.

Initial simple approach:

- Use deterministic random splitting with a seed.

Possible stronger approach:

- Split by time ranges so test windows come from a held-out segment of the signal.

Decision for baseline:

- Start with deterministic random 80/20 split.
- Document that time-based splitting may be tested later for a stricter evaluation.

Acceptance criteria:

- Train and test sizes match the configured ratio.
- Splits are reproducible with the same seed.
- Tests verify split sizes and no duplicate indices between train and test.

### Phase 5: Model Definitions

Goals:

- Implement three model architectures with consistent outputs.

Implementation file:

- `src/ai_orchestra/models.py`

Models:

- `FullyConnectedSignalNet`
- `RnnSignalNet`
- `LstmSignalNet`

Fully connected model:

- Input: `(batch_size, 14)`
- Hidden layers: configurable
- Activation: ReLU
- Output: `(batch_size, 10)`

Simple RNN model:

- Input: `(batch_size, 10, 5)`
- Recurrent hidden size: configurable
- Number of layers: configurable
- Output: `(batch_size, 10)`

LSTM model:

- Input: `(batch_size, 10, 5)`
- Hidden size: configurable
- Number of layers: configurable
- Output: `(batch_size, 10)`

Output strategy for recurrent models:

- Use recurrent outputs across the full sequence.
- Map the recurrent output at each time step to one predicted signal value.
- Squeeze the final feature dimension to produce a 10-value window.

Acceptance criteria:

- Each model produces output shape `(batch_size, 10)`.
- Unit tests verify model forward passes with fake tensors.
- Model creation is driven by config values.

### Phase 6: Training Loop

Goals:

- Train any selected model using the same training pipeline.
- Save metrics and optionally save model checkpoints.

Implementation file:

- `src/ai_orchestra/training.py`

Training behavior:

- Use MSE loss.
- Use Adam optimizer by default.
- Track train loss per epoch.
- Track validation/test loss per epoch.
- Support CPU by default.
- Use GPU automatically if available, but do not require it.
- Use deterministic seeds where possible.

Training outputs:

- Final trained model
- Loss history
- Final test loss
- Optional checkpoint under `results/models/`

Acceptance criteria:

- A tiny smoke training run completes quickly.
- Loss history contains one value per epoch.
- Training works for all three model types.
- Integration test trains each model for a very small number of epochs and confirms no crash.

### Phase 7: Evaluation

Goals:

- Compare models using numeric metrics and prediction samples.

Implementation file:

- `src/ai_orchestra/evaluation.py`

Metrics:

- MSE
- Optional MAE
- Optional per-frequency MSE

Evaluation outputs:

- Overall metrics for each architecture.
- Metrics grouped by requested frequency.
- Example predictions for plotting.

Acceptance criteria:

- Evaluation returns a structured dictionary.
- Results can be saved as JSON or CSV under `results/metrics/`.
- Evaluation can compare FC, RNN, and LSTM results in one place.

### Phase 8: Plotting

Goals:

- Generate clear visualizations for the README and final comparison.

Implementation file:

- `src/ai_orchestra/plotting.py`

Required plots:

- Clean individual signals.
- Noisy individual signals.
- Clean mixed signal.
- Noisy mixed signal.
- Training and validation loss curves.
- True clean window vs. predicted window for each model.
- Model comparison plot using final test loss.

Optional plots:

- Per-frequency error comparison.
- Noise-level experiment comparison.
- Window-size experiment comparison.

Output directory:

- `results/figures/`

Acceptance criteria:

- Plot functions save images without requiring an interactive display.
- Plot filenames are predictable.
- README can reference the generated images.

### Phase 9: Experiment Runner

Goals:

- Run the full pipeline from config.
- Train and compare all three architectures.
- Save metrics and plots.

Implementation file:

- `src/ai_orchestra/experiments.py`

SDK entry:

- `src/ai_orchestra/sdk.py`

Main SDK functions:

- `generate_data(config_path=None)`
- `prepare_datasets(config_path=None)`
- `train_model(model_name, config_path=None)`
- `run_baseline_experiment(config_path=None)`
- `run_comparison_experiment(config_path=None)`

Baseline run:

- Generate one signal set.
- Build datasets.
- Train FC, RNN, and LSTM.
- Evaluate each model.
- Save metrics.
- Save plots.

Acceptance criteria:

- One command or SDK call runs the baseline experiment.
- Outputs are written to `results/`.
- Results are reproducible with the configured seed.

### Phase 10: README Documentation

Goals:

- Explain the project clearly.
- Show how to install, run, test, and interpret results.

README sections:

- Project overview
- Problem statement
- Signal-generation explanation
- Dataset format
- Model architectures
- Training process
- How to run the baseline experiment
- How to run tests
- Example plots
- Experiment observations
- Conclusions

README should include:

- Screenshots or linked generated figures.
- Training results.
- Model prediction examples.
- Comparison of FC, RNN, and LSTM.
- Notes on what changed when varying noise, hidden neurons, layers, or window size.

Acceptance criteria:

- A new user can run the project from the README.
- README commands use `uv`.
- README conclusions match actual generated results.

## 7. Testing Plan

Use tests to protect the project from silent ML/data bugs.

Unit tests:

- Signal generation returns correct shapes.
- Signal generation is reproducible with a seed.
- Dataset length is correct.
- One-hot vectors map to correct target signals.
- Window slicing is correct.
- FC dataset item shape is correct.
- Sequence dataset item shape is correct.
- Each model produces the expected output shape.

Integration tests:

- Baseline data generation plus dataset preparation works.
- Each model can run one tiny training loop.
- Evaluation returns valid metrics.

Quality checks:

- `uv run pytest`
- `uv run ruff check .`

Coverage goal:

- Aim for at least 85% coverage on meaningful non-plotting code.

## 8. Experiment Plan

Start with a baseline experiment:

- Frequencies: `[1, 3, 5, 7]`
- Samples: `10000`
- Sampling frequency: `1000 Hz`
- Window size: `10`
- Split: `80/20`
- Models: FC, RNN, LSTM
- Same dataset for all models
- Same number of epochs where practical

Then run controlled variants:

- Increase and decrease noise level.
- Try different hidden sizes.
- Try different numbers of layers.
- Try different window sizes.
- Optionally compare `70/30`, `80/20`, and `90/10` splits.

Keep experiments fair:

- Use the same generated data where possible.
- Use the same random seed when comparing architectures.
- Do not compare models trained with different data unless the experiment is explicitly about data.
- Record configuration values with each result.

## 9. Risks And Mitigations

Risk: Dataset is too easy because the signals are simple and fixed.

Mitigation:

- Add more random amplitude, phase, and noise variation.
- Generate multiple signal sets.
- Evaluate on held-out generated signals.

Risk: Random window splitting leaks similar neighboring windows into train and test sets.

Mitigation:

- Start with random split for speed.
- Add time-based split for stricter evaluation.

Risk: Dataset becomes very large when many noise variants are generated.

Mitigation:

- Generate data lazily if needed.
- Keep baseline small.
- Add config controls for number of generated signal sets.

Risk: RNN/LSTM comparison may be affected by model size rather than architecture.

Mitigation:

- Track parameter counts.
- Use comparable hidden sizes.
- Document architecture differences clearly.

Risk: Training may take too long on CPU.

Mitigation:

- Use small defaults first.
- Make epochs and hidden sizes configurable.
- Add smoke-test configs for quick validation.

## 10. Work Order

Follow this order:

1. Create project setup and package structure.
2. Add config loading.
3. Implement signal generation.
4. Test signal generation.
5. Implement sliding-window dataset.
6. Test dataset indexing and shapes.
7. Implement FC, RNN, and LSTM models.
8. Test model forward passes.
9. Implement training loop.
10. Add smoke training tests.
11. Implement evaluation metrics.
12. Implement plotting utilities.
13. Implement baseline experiment runner.
14. Run baseline experiment.
15. Save figures and metrics.
16. Update README with real results.
17. Add experiment variants.
18. Update conclusions after comparing results.

## 11. Definition Of Done

The project is complete when:

- The repo has a clean package structure.
- Signals are generated from configurable parameters.
- Sliding-window datasets are created correctly.
- FC, RNN, and LSTM models all train successfully.
- All three models are evaluated with comparable metrics.
- Plots are generated and saved.
- README explains the project, setup, usage, results, and conclusions.
- Tests pass.
- Ruff passes.
- No secrets or generated heavy artifacts are committed accidentally.
- The implementation follows the plan unless we intentionally update the plan.

