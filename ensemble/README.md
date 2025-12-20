
#ABM-UDE MS+PEM Ensemble (Julia)
This repo runs a multi-process (multi-core) ensemble of **MS+PEM** training runs
 (e.g., 100 seeds) and writes per-seed outputs (metrics + time series), plus scr
ipts to aggregate results for plotting.

## Requirements
- **Julia 1.10.10** (recommended)
- Linux/Ubuntu (tested on Windows also)
- Enough RAM for multi-process runs (`-p` workers are full Julia processes)

## Expected repo layout 
── julia_test/
│ ├── Project.toml
│ └── Manifest.toml
├── train_pem_ms_ensemble_multicore.jl
├── utils_1NN.jl
├── avg_output.dat
└── (optional scripts)
├── aggregate_ms_pem_ensemble.jl
└── plot_ensemble_final.jl



Note: the training script activates `julia_test/` as its environment, so both th
e master and worker processes use the same dependencies.

## 1) Install dependencies (one-time)
From the repo root:
julia --project=julia_test -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

#pick 8 workers
julia --project=julia_test --threads=1 -p 8 train_pem_ms_ensemble_multicore.jl

#auto pick workers
julia --project=julia_test --threads=1 -p auto train_pem_ms_ensemble_multicore.jl

