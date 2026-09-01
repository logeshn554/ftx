"""Custom PyTorch policy networks for the PPO agent.

Provides feature extractors that handle the 2D observation space
(window × features) using either MLP (flattened), 1D-CNN, or LSTM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MLPFeatureExtractor(BaseFeaturesExtractor):
    """Flatten the 2D observation and pass through an MLP.

    Simple but effective for smaller observation spaces.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        hidden_dim: int = 256,
        features_dim: int | None = None,
    ) -> None:
        if features_dim is not None:
            hidden_dim = features_dim
        flat_dim = int(observation_space.shape[0] * observation_space.shape[1])
        features_dim = hidden_dim

        super().__init__(observation_space, features_dim)

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """1D-CNN over the temporal window of features.

    Treats the feature window as a 1D signal with n_features channels.
    Good for capturing local temporal patterns.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[1]  # n_features + portfolio

        self.cnn = nn.Sequential(
            # (batch, window, channels) → (batch, channels, window) for Conv1d
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch, window, features)
        x = observations.permute(0, 2, 1)  # → (batch, features, window)
        x = self.cnn(x)
        x = x.squeeze(-1)  # → (batch, 64)
        return self.fc(x)


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """LSTM over the temporal window for sequential pattern recognition.

    Best for capturing long-range temporal dependencies in the feature window.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_features = observation_space.shape[1]

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, window, features)
        lstm_out, _ = self.lstm(observations)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        return self.fc(last_hidden)


# Registry of feature extractors
FEATURE_EXTRACTORS = {
    "mlp": MLPFeatureExtractor,
    "cnn": CNNFeatureExtractor,
    "lstm": LSTMFeatureExtractor,
}
