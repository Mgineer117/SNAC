import numpy as np


class ObservationNormalizer:
    def __init__(
        self,
        state_dim,
        mode="ema",
        alpha=0.95,
        max_updates=10000,
    ):
        """
        mode: Type of moving average ('ema', 'cma', or 'wma')
        alpha: Decay factor for EMA
        window_size: Size of the window for WMA
        weights: Custom weights for WMA
        max_updates: Maximum number of updates allowed for normalization
        state_dim: The state dimension (shape of the observation, e.g., (batch_size, height, width, channels))
        """
        self.mode = mode
        self.alpha = alpha
        self.max_updates = max_updates  # Maximum number of updates allowed
        self.update_count = 0  # Track the number of updates

        # Initialize the statistics
        self.count = 0
        self.eps = 1e-2

        # Initialize based on the mode
        if mode == "ema":
            self.normalizer = EMA(alpha=self.alpha, state_dim=state_dim)
        elif mode == "cma":
            self.normalizer = CMA(state_dim=state_dim)
        elif mode == "none":
            self.normalizer = None
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented.")

    def normalize(self, observation, update=True):
        """
        Normalize the observation using the selected moving average mode.
        If the observation is a batch, apply normalization per observation.
        """
        # Stop updating once the max updates are reached
        if self.update_count >= self.max_updates:
            return (observation - self.mean) / (np.sqrt(self.var) + self.eps)
        else:
            if update:
                self.normalizer.update(observation)
            # Update the statistics based on the selected mode
            self.mean = self.normalizer.mean
            if self.mode == "ema":
                self.var = self.normalizer.variance
            elif self.mode == "cma":
                # CMA doesn't maintain variance
                self.var = np.zeros_like(self.mean)

            self.update_count += 1

            # Return the normalized observation after the update
            return (observation - self.mean) / (np.sqrt(self.var) + self.eps)


class EMA:
    def __init__(self, alpha=0.99, state_dim=None):
        """
        alpha: Decay factor for the exponential moving average.
        state_dim: The state dimension, e.g., (batch_size, height, width, channels)
        """
        self.alpha = alpha
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim)  # Initialize mean based on state dimension
        self.variance = np.ones(state_dim)  # Initialize variance with ones

    def update(self, observation):
        """
        Update the moving averages of mean and variance using the EMA formula.
        """
        if self.mean is None:
            self.mean = observation
            self.variance = np.zeros_like(observation)
        else:
            # Update mean and variance using EMA formula
            self.mean = self.alpha * self.mean + (1 - self.alpha) * observation
            self.variance = (
                self.alpha * self.variance
                + (1 - self.alpha) * (observation - self.mean) ** 2
            )


class CMA:
    def __init__(self, state_dim=None):
        """
        Cumulative moving average (CMA) for mean calculation.
        state_dim: The state dimension (shape of the observation, e.g., (batch_size, height, width, channels))
        """
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim)  # Initialize mean based on state dimension
        self.count = 0

    def update(self, observation):
        """
        Update the cumulative moving average of the observations.
        """
        self.count += 1
        if self.mean is None:
            self.mean = observation
        else:
            # Update cumulative mean using CMA formula
            self.mean = (self.mean * (self.count - 1) + observation) / self.count
