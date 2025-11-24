# abm-ude

## ABM-UDE Surrogate (κ-Neural Parameterization) — Julia Implementation

This repository contains the full Julia implementation of the Universal Differential Equation (UDE) surrogate used to learn epidemic contact-rate dynamics from ExaEpi agent-based model trajectories.

The surrogate model replaces the classical contact-rate parameter κ(t) with a single-output neural network, embedded inside the mechanistic SEInsIsIaDR ODE system. Training is stabilized using multiple shooting and an observer-based Prediction Error Method (PEM).

Repository Structure:
```
.
├── julia_test/                    # Local Julia project environment (Project.toml, Manifest.toml)
│
├── ODE_models_1NN.jl             # Mechanistic SEInsIsIaDR model + 1-neuron-output UDE (κϕ)
│
├── utils_1NN.jl                  # Data processing, normalization, MS/PEM utilities, loss functions
│
├── train_ude_multiple_shooting.jl# Train UDE using Multiple Shooting (MS-UDE)
│
├── train_ude_pem.jl              # Train UDE using Prediction Error Method (PEM-UDE)
│
├── pem_ude_kappa_ensemble.jl     # Script for ensemble inference + plotting κϕ(t) behavior
│
├── avg_output.dat                # Example ExaEpi averaged trajectory (training data)
│
└── README.md                     # This file
```

