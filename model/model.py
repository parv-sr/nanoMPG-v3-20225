import torch
import torch.nn as nn


class MPGModel(nn.Module):
    def __init__(self):
        super(MPGModel, self).__init__()

        self.fc1 = nn.Linear(
            in_features=5,
            out_features=6
        )

        self.fc2 = nn.Linear(
            in_features=6,
            out_features=5
        )

        self.fc3 = nn.Linear(
            in_features=5,
            out_features=5
        )

        self.fc4 = nn.Linear(
            in_features=5,
            out_features=4
        )

        self.fc5 = nn.Linear(
            in_features=4,
            out_features=3
        )

        self.fc6 = nn.Linear(
            in_features=3,
            out_features=1
        )

        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Hidden Layer 1
        x = self.fc1(x)
        x = self.activation(x)

        # Hidden Layer 2
        x = self.fc2(x)
        x = self.activation(x)

        # Hidden Layer 3
        x = self.fc3(x)
        x = self.activation(x)

        # Hidden Layer 4
        x = self.fc4(x)
        x = self.activation(x)

        # Hidden Layer 5
        x = self.fc5(x)
        x = self.activation(x)

        # Output Layer
        x = self.fc6(x)
