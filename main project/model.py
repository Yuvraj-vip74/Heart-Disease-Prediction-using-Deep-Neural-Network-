# model.py
# PURPOSE: Define the Deep Neural Network architecture for heart disease prediction.
#
# Architecture:
#   INPUT  → 15 features (8 numeric/binary + 4 cp one-hot + 3 restecg one-hot)
#   Layer 1: 128 neurons  → ReLU + BatchNorm + Dropout(0.4)
#   Layer 2: 64  neurons  → ReLU + BatchNorm + Dropout(0.3)
#   Layer 3: 32  neurons  → ReLU + BatchNorm + Dropout(0.2)
#   OUTPUT : 2 neurons    → Class 0: No Disease | Class 1: Disease

import torch
import torch.nn as nn


class HeartNet(nn.Module):
    """Feedforward DNN for binary heart-disease classification."""

    def __init__(self, input_size: int):
        super(HeartNet, self).__init__()

        self.network = nn.Sequential(
            # ── Hidden Layer 1 ────────────────────────────────
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # ── Hidden Layer 2 ────────────────────────────────
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # ── Hidden Layer 3 ────────────────────────────────
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # ── Output Layer ──────────────────────────────────
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == '__main__':
    # Quick sanity check
    model = HeartNet(input_size=15)
    dummy = torch.randn(8, 15)
    out   = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")
    print("Model OK ✅")
