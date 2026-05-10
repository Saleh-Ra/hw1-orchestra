# Requirements

## Project Goal

Build a machine learning project that learns how to retrieve one specific clean sine signal from a noisy mixed signal.

The project is based on signal processing and neural networks. Several sine waves with different frequencies are combined into one mixed signal, noise is added, and a neural network receives:

- A small noisy window from the mixed signal.
- A one-hot encoded vector that tells the network which frequency to retrieve.

The network should output the clean version of the requested signal window.

## Signal Generation

Generate four sine signals:

- `S1`: 1 Hz
- `S2`: 3 Hz
- `S3`: 5 Hz
- `S4`: 7 Hz

Signal generation requirements:

- Sampling frequency: `1000 Hz`
- Number of samples per signal: `10000`
- Total duration: `10 seconds`
- Each sine signal should include random amplitude and random phase.
- Noise should be added to amplitude and phase to create noisy signal versions.
- The formula does not need to be mathematically perfect, as long as each signal includes random amplitude noise and phase noise.
- `beta` should be between `0` and `2pi`.

For every generated sample set, create:

- Clean signals: `S1`, `S2`, `S3`, and `S4`
- Noisy versions of each signal
- One clean combined signal
- One noisy combined signal

The noisy combined signal is the main input source for the neural networks.

## Dataset Preparation

Create a large dataset using sliding windows.

- Window size: `10` samples
- For every position in the signal, extract a 10-sample window from the noisy combined signal.
- At the same position, extract the matching 10-sample clean window from one of the clean signals.
- The selected clean signal depends on the one-hot encoded vector `C`.

The one-hot encoded vector `C` has length `4`:

- `[1, 0, 0, 0]` retrieves `S1`, the `1 Hz` signal.
- `[0, 1, 0, 0]` retrieves `S2`, the `3 Hz` signal.
- `[0, 0, 1, 0]` retrieves `S3`, the `5 Hz` signal.
- `[0, 0, 0, 1]` retrieves `S4`, the `7 Hz` signal.

Each training sample contains:

- Input: the one-hot vector `C` together with the noisy mixed signal window.
- Target: the clean signal window corresponding to the selected frequency.

Example:

If `C = [0, 1, 0, 0]` and the noisy mixed signal window uses samples `6203` to `6213`, then the target should be samples `6203` to `6213` from the clean `S2` signal.

## Models To Compare

Implement and compare three neural network architectures:

- Fully connected neural network
- Simple RNN
- LSTM

The goal is to compare how well each architecture reconstructs the requested clean signal from the noisy combined signal.

### Fully Connected Model

- Input: flat vector containing the 10 noisy samples plus the 4 values from `C`
- Input size: `14`
- Output: 10 predicted values representing the reconstructed clean signal window

### RNN And LSTM Models

- Input: sequence of length `10`
- Each time step contains:
  - The current noisy sample
  - The full one-hot vector `C`
- Features per time step: `5`
- Output: 10 predicted values representing the reconstructed clean signal window

## Training And Evaluation

Use Mean Squared Error between the predicted signal window and the true clean signal window as the loss function.

Train all three models and compare performance using:

- Training loss
- Validation loss
- Prediction quality

## Visualizations

Generate plots and visualizations showing:

- Example clean signals
- Example noisy signals
- Clean and noisy mixed signals
- True clean signal window vs. predicted signal window for each model
- Training loss curves for the different models
- Comparison graphs or tables showing which architecture performs better under different conditions

## Experiments And Observations

The README should explain the experiments and observations. Possible experiments include changing:

- Noise level
- Number of hidden neurons
- Number of layers
- Window size

The README should include:

- Screenshots of graphs
- Training results
- Model predictions
- Conclusions about which model performed best and why

Possible conclusions may include:

- LSTM handles sequential information better than a fully connected network.
- RNN performance changes depending on noise level and sequence length.

## Project Organization

Organize the project cleanly into multiple files, such as:

- Data generation
- Dataset preparation
- Model definitions
- Training
- Evaluation
- Plotting utilities
- README documentation

## Open Questions

- Should the project use PyTorch, TensorFlow/Keras, or another framework?
- Should generated datasets be saved to disk, regenerated each run, or both?
- Should experiments be controlled from a config file?
- What train/validation/test split should we use?
- Which plots should be committed as example results?
