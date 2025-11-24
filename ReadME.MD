# Universal Differential Equation Training (Julia)

This repository contains a small Julia codebase for experimenting with **Universal Differential Equations (UDEs)** and **multiple shooting** for parameter / neural network training on ODE models.

The current setup uses a 1‑hidden‑layer neural network (see `ODE_models_1NN.jl` and `utils_1NN.jl`) and supports two main training pipelines:

- **Prediction Error Method (PEM)** single‑shooting training
- **Multiple‑shooting UDE training**

All scripts are written in Julia and use the project environment stored in the `julia_test/` folder.

---

## Repository Structure

- `julia_test/`  
  Julia project environment (contains `Project.toml` and `Manifest.toml`). All scripts activate this environment.

- `ODE_models_1NN.jl`  
  ODE / UDE model definitions that use a 1‑hidden‑layer neural network. This is where the dynamical system and NN architecture are wired together.

- `utils_1NN.jl`  
  Utility functions for:
  - Loading and preprocessing data
  - Building neural networks and parameter containers
  - Defining loss functions
  - Running optimization loops and plotting helpers

- `avg_output.dat`  
  Example dataset used as ground‑truth observations for PEM and multiple‑shooting training. Typically contains averaged time‑series output (e.g., compartment trajectories) in a simple text / matrix format.

- `train_ude_pem.jl`  
  Script that trains the UDE with a **single‑shooting Prediction Error Method (PEM)** using `avg_output.dat` as input data. Good starting point for understanding the basic training loop.

- `train_ude_multiple_shooting.jl`  
  Script that trains the same (or similar) UDE using **multiple shooting**. The time horizon is split into segments, each with its own initial condition, plus continuity penalties at segment boundaries for stability.

- `pem_ude_kappa_ensemble.jl`  
  Script for running **ensemble PEM experiments** (e.g., sweeping over different κ / hyperparameters or random seeds) to study robustness and variability of the trained UDE.

---

## Getting Started

### 1. Clone the repository

```bash
git clone <this-repo-url>.git
cd <this-repo-folder>
```

### 2. Install Julia dependencies

All dependencies are managed through the `julia_test` environment.

From the repository root, start Julia and instantiate the environment:

```julia
using Pkg
Pkg.activate("julia_test")
Pkg.instantiate()
```

Alternatively, from the command line you can launch Julia with the project directly:

```bash
julia --project=julia_test
```

and then run

```julia
using Pkg
Pkg.instantiate()
```

This will download all required packages (e.g., `OrdinaryDiffEq`, `SciMLSensitivity`, `Optimization`, `Lux`, `ComponentArrays`, `Plots`, etc.).

---

## Running the Training Scripts

> **Tip:** All commands below assume you are in the repository root and that the `julia_test` environment has been instantiated at least once.

### 1. PEM (single‑shooting) training

Runs single‑shooting UDE training using Prediction Error Method on `avg_output.dat`.

```bash
julia --project=julia_test train_ude_pem.jl
```

What it does (conceptually):

- Loads `avg_output.dat` as the observation data
- Builds the ODE/UDE model and 1‑hidden‑layer NN
- Defines a prediction‑error loss between simulated trajectories and data
- Optimizes network parameters (and possibly ODE parameters) using ADAM / LBFGS
- Prints training loss to the console and saves plots / outputs as defined in the script

### 2. Multiple‑shooting UDE training

Runs the multiple‑shooting training pipeline:

```bash
julia --project=julia_test train_ude_multiple_shooting.jl
```

Conceptually this script:

- Splits the overall time interval into several segments
- Treats the initial condition of each segment as an optimization variable
- Adds continuity penalties between segments to enforce a smooth global trajectory
- Optimizes the neural network and (optionally) ODE parameters for better stability and data fit
- Produces logs, plots, and any artifacts configured inside the script

### 3. PEM κ‑ensemble experiments

To explore robustness across different κ values, random seeds, or run configurations, use:

```bash
julia --project=julia_test pem_ude_kappa_ensemble.jl
```

The script typically:

- Loops over a set of κ / hyperparameter choices
- Runs PEM training for each configuration
- Aggregates or saves results (e.g., final losses, plots, or parameter snapshots)

Check the comments at the top of `pem_ude_kappa_ensemble.jl` for the exact set of experiments and output locations.

---

## Data

The repository currently expects an input file

- `avg_output.dat` – example dataset used for training and evaluation.

If you want to use your own data:

1. Match the data format expected in `utils_1NN.jl` (shape, order of states, time spacing).  
2. Update the path or loader function in the relevant training script.  
3. Re‑run `train_ude_pem.jl` or `train_ude_multiple_shooting.jl`.

---

## Reproducibility

- Training scripts typically set random seeds via `StableRNGs` for reproducible runs.
- If you change the neural network architecture or optimizer settings in `utils_1NN.jl`, document those changes for future comparison.
- For ensemble runs (`pem_ude_kappa_ensemble.jl`), you can extend the loop over seeds or κ values to generate more robust statistics.

---

## Customization

A few natural places to modify the code:

- **Model / NN architecture** – adjust the layers and activations in `ODE_models_1NN.jl` and `utils_1NN.jl`.
- **Loss function** – tweak the weighting of states, log‑scale vs linear loss, or train/test splits in `utils_1NN.jl`.
- **Training schedule** – change number of ADAM / LBFGS iterations, learning rate, or multiple‑shooting segment lengths in the training scripts.

This repo is intended as a flexible sandbox for experimenting with **SciML + UDEs + multiple shooting**, so feel free to adapt it to new models or datasets.

---

## License

No explicit license has been added yet. If you plan to use this code outside of personal or internal research, please add an appropriate LICENSE file or contact the author to clarify usage rights.
