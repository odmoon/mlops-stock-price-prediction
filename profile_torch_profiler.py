# Profiling Step 4 of Part 2 of Project
#by Maheen Khan
# new profiling script using torch.profiler

import torch
import torch.profiler
import os
import sys

# Dummy PyTorch model for demonstration
import torch.nn as nn
import torch.optim as optim

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.lstm = nn.LSTM(4, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Function to simulate training (replace with actual PyTorch training loop)
def train_model():
    model = PyTorchModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.randn(100, 60, 4)
    targets = torch.randn(100, 1)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Profile the training process
def profile_training():
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/torch_profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        train_model()
        prof.step()

if __name__ == "__main__":
    profile_training()
    print("Torch profiler results saved to ./log/torch_profiler")
