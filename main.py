#!/usr/bin/env python

import os
import numpy as np
import torch

from src.dynamics import Dynamics
from src.trainer import Trainer

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data
dataset = "Angle"
data = np.loadtxt(os.path.join('data', '{}.csv'.format(dataset)))

# State (pos,vel)
X = torch.from_numpy(data[:, :4]).float().to(device)  # .requires_grad_(True)

# Output (acc)
Y = torch.from_numpy(data[:, 4:6]).float().to(device)  # .requires_grad_(True)

# Create Model
dim = 2
structure = [100, 100]
attractor = X[-1, :dim]
K = torch.eye(dim, dim).to(device)
D = 4*torch.eye(dim, dim).to(device)

ds = Dynamics(2, attractor, structure).to(device)
ds.dissipation = (D, False)
ds.stiffness = (K, False)

# Create trainer
trainer = Trainer(ds, X, Y)

# Set trainer optimizer (this is not very clean)
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(), lr=1e-2,  weight_decay=1e-8)

# Set trainer loss
trainer.loss = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()

# Set trainer options
trainer.options(normalize=False, shuffle=True, print_loss=True,
                epochs=1000, load_model=None)  # None, dataset

# Train model
trainer.train()

# Save model
trainer.save(dataset)