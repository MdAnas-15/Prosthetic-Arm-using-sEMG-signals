import torch
from torch import nn


class CNNGRU(nn.Module):
    def __init__(self, channels: int, num_classes: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 96, 5, padding=2),
            nn.BatchNorm1d(96), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(96, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        return self.head(x[:, -1])
