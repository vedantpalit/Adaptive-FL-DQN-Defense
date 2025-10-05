import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleCNN(nn.Module):
    """A lightweight CNN for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 8 * 8)
        return self.fc(x)

class BackdoorTrigger:
    """Implements various backdoor trigger patterns."""
    def __init__(self, trigger_type='pixel', trigger_size=4, target_label=0):
        self.trigger_type = trigger_type
        self.trigger_size = trigger_size
        self.target_label = target_label

    def apply_trigger(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, H, W)
        triggered = x.clone()
        if self.trigger_type == 'pixel':
            triggered[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        elif self.trigger_type == 'pattern':
            for i in range(self.trigger_size):
                for j in range(self.trigger_size):
                    if (i + j) % 2 == 0:
                        triggered[:, -(i+1), -(j+1)] = 1.0
        elif self.trigger_type == 'distributed':
            np.random.seed(123)
            positions = np.random.choice(32*32, self.trigger_size, replace=False)
            for pos in positions:
                r, c = pos // 32, pos % 32
                triggered[:, r, c] = 1.0
        return triggered
