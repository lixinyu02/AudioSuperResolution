import torch
from torch import nn

class SpectralSuperResolution(nn.Module):
    def __init__(self):
        super(SpectralSuperResolution, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
