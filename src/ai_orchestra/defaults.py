"""Default values for the first minimal reconstruction pipeline."""

from dataclasses import dataclass
from math import tau


@dataclass(frozen=True)
class PipelineDefaults:
    """Code defaults used before the project grows a full config layer."""

    frequencies_hz: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0)
    sampling_frequency_hz: int = 1000
    num_samples: int = 10_000
    duration_seconds: float = 10.0
    window_size: int = 10
    num_signal_sets: int = 50
    train_ratio: float = 0.8
    batch_size: int = 64
    random_seed: int = 42
    amplitude_range: tuple[float, float] = (0.5, 1.5)
    phase_range: tuple[float, float] = (0.0, tau)
    amplitude_noise_std: float = 0.10
    phase_noise_std: float = 0.10
    additive_noise_std: float = 0.02

    def validate(self) -> None:
        """Raise a clear error if a default value is internally inconsistent."""
        if not self.frequencies_hz:
            msg = "At least one frequency is required."
            raise ValueError(msg)
        if self.sampling_frequency_hz <= 0:
            msg = "Sampling frequency must be positive."
            raise ValueError(msg)
        if self.num_samples <= 0:
            msg = "Number of samples must be positive."
            raise ValueError(msg)
        if self.window_size <= 0 or self.window_size > self.num_samples:
            msg = "Window size must be positive and no larger than num_samples."
            raise ValueError(msg)
        if self.num_signal_sets <= 0:
            msg = "At least one signal set is required."
            raise ValueError(msg)
        if not 0.0 < self.train_ratio < 1.0:
            msg = "Train ratio must be between 0 and 1."
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "Batch size must be positive."
            raise ValueError(msg)
        if (
            self.amplitude_range[0] <= 0
            or self.amplitude_range[0] >= self.amplitude_range[1]
        ):
            msg = "Amplitude range must be positive and increasing."
            raise ValueError(msg)
        if self.phase_range != (0.0, tau):
            msg = "Default phase range should stay from 0 to 2pi."
            raise ValueError(msg)
        if min(
            self.amplitude_noise_std,
            self.phase_noise_std,
            self.additive_noise_std,
        ) < 0:
            msg = "Noise strengths must be non-negative."
            raise ValueError(msg)


DEFAULTS = PipelineDefaults()
