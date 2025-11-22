cd(dirname(@__FILE__))
using Pkg
Pkg.activate("julia_test")

using MAT, JLD2, Plots, Measures, Random, LaTeXStrings
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq
using LinearAlgebra, DiffEqFlux, Optim
using SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Zygote, StableRNGs, Statistics
using CSV, DataFrames
using ReverseDiff
using LineSearches
using Colors
using Plots

model_architecture = "_1NN"   # kept for utils/models includes if you reuse them
utils_adjust = ""
method = model_architecture * utils_adjust

utils_path = "utils" * method * ".jl"
models_path = "ODE_models" * model_architecture * ".jl"

ADAM_maxiters = 200000
Weights_dir_name = "Weights" * method * "_kappa_only/"

include(utils_path)     # provides process_data(...)
# include(models_path)  # not needed for PEM-only variant
if !isdir(Weights_dir_name) mkdir(Weights_dir_name) end

model = "SEInsIsIaDR"
est_all = false
scale_NN = true
num_inputs = [6]

df = DataFrame(CSV.File("avg_output.dat", delim=' ', ignorerepeated=true))
println("Columns now: ", names(df))

selected_df, tdata, tspan, train_n_days, ode_data, u0, max_vals = process_data(df, model)

S, T = size(ode_data)
@assert S == 6
const STATE_NAMES = ["Never", "Exposed", "Presymptomatic", "Symptomatic", "Asymptomatic", "Deaths"]

function make_linear_interpolator(tgrid::AbstractVector, Y::AbstractMatrix)
    @assert size(Y, 2) == length(tgrid)
    cols = [@view Y[:, i] for i in 1:length(tgrid)]
    function y_of_t(t::Real)
        if t <= tgrid[1]; return copy(cols[1]); end
        if t >= tgrid[end]; return copy(cols[end]); end
        i = searchsortedlast(tgrid, t) |> x -> clamp(x, 1, length(tgrid)-1)
        t0 = tgrid[i]; t1 = tgrid[i+1]
        w = (t - t0) / (t1 - t0)
        @. (1 - w) * cols[i] + w * cols[i+1]
    end
    return y_of_t
end

y_of_t = make_linear_interpolator(tdata, ode_data)

softplus(x) = log1p(exp(x))

nn_kappa = Chain(
    Dense(6, 32, tanh),
    Dense(32, 16, tanh),
    Dense(16, 1) # softplus applied outside to keep AD stable
)
p_init, st = Lux.setup(Xoshiro(0), nn_kappa)

# Parameters: just θ (no per-state K)
p0 = ComponentArray(; θ = p_init)

# Optional: pick which states to nudge (1=on, 0=off)
const OBS_MASK = ones(Float64, S)  # set zeros where you don't want nudging

function pem_predictor(u, p, t)
    κ_raw = nn_kappa(u, p.θ, st)[1][1]       # scalar
    κ = softplus(κ_raw) + 1e-6               # ensure strictly positive, avoid zero
    return κ .* (OBS_MASK .* (y_of_t(t) .- u))
end

prob_pred = ODEProblem((du,u,p,t)->(du .= pem_predictor(u,p,t)), u0, tspan, p0, saveat=tdata)

ps = p0
pd, pax = getdata(ps), getaxes(ps)

global y = ode_data  # normalized data S×T

function predloss(pvec)
    p_struct = ComponentArray(pvec, pax)
    yh = solve(remake(prob_pred, p = p_struct), Tsit5(); saveat = tdata)
    if !SciMLBase.successful_retcode(yh.retcode)
        return Inf
    end
    pred = Array(yh)                # S×T
    e2 = abs2.(y .- pred)
    return mean(e2)
end

adtype = Optimization.AutoZygote()
optf   = Optimization.OptimizationFunction((x, p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pd)

mkpath("PEM_KAPPA_ONLY")
loss_history = Float64[]
iter = 0

function save_predictions_csv(
    sol_arr::AbstractArray,          # S×T
    tdata::AbstractVector,           # length T
    ode_data::AbstractArray;         # S×T
    outdir::AbstractString = "PEM_KAPPA_ONLY_csv",
    tag::AbstractString = ""
)
    mkpath(outdir)
    S, T = size(sol_arr)
    cols = Dict{Symbol, Any}()
    cols[:time] = collect(tdata)
    @inbounds for (i, nm) in enumerate(STATE_NAMES)
        cols[Symbol("$(nm)_obs")]  = vec(ode_data[i, :])
        cols[Symbol("$(nm)_pred")] = vec(sol_arr[i, :])
    end
    df_out = DataFrame(cols)
    fname  = isempty(tag) ? "predictions.csv" : "predictions_$(tag).csv"
    CSV.write(joinpath(outdir, fname), df_out)
    println("💾 Saved predictions CSV → ", joinpath(outdir, fname))
end

function plot_u0_fitting_periodic(sol, tdata, ode_data, u0, iter; max_vals=nothing, lockdown_time=nothing)
    mkpath("PEM_KAPPA_ONLY")
    sol_arr  = sol isa AbstractArray ? sol : Array(sol)
    data_arr = ode_data
    nstates = length(STATE_NAMES)
    for i in 1:nstates
        state = STATE_NAMES[i]
        plt = plot(
            title = "PEM κ-only: $state Fit",
            xlabel = "Time", ylabel = "Population",
            legend = :topleft, size = (800, 500),
            titlefontsize = 14, guidefontsize = 12, tickfontsize = 10
        )
        scatter!(plt, tdata, data_arr[i, :], label = "Observed", markershape = :circle, color = :black, markersize = 5)
        plot!(plt, tdata, sol_arr[i, :], label = "Predicted", linestyle = :solid, linewidth = 2, color = :red)
        savefig(plt, "PEM_KAPPA_ONLY/u0_fitting_$(replace(state, ' '=>'_'))_iter_$(iter).png")
        display(plt)
    end
end

callback = function (p, l; doplot = true)
    global iter, loss_history
    iter += 1
    push!(loss_history, l)

    if doplot && iter % 1000 == 0
        p_current = ComponentArray(p.u, pax)
        sol = solve(prob_pred, Tsit5(); p = p_current, saveat = tdata)

        denorm_ode_data = ode_data .* max_vals
        denorm_sol      = sol .* max_vals

        save_predictions_csv(denorm_sol, tdata, denorm_ode_data;
            outdir = "PEM_KAPPA_ONLY_csv",
            tag   = "iter_$(lpad(iter, 6, '0'))")

        plot_u0_fitting_periodic(denorm_sol, tdata, denorm_ode_data, u0, iter)
    end
    return false
end

res_adam = Optimization.solve(optprob, ADAM(0.01); callback = callback, maxiters = ADAM_maxiters)
adam_loss = loss_history[end]
println("ADAM Final Loss: ", adam_loss)

p_LBFGS_init   = getdata(ComponentArray(res_adam.u, pax))
loss_history   = Float64[]
optprob_LBFGS  = Optimization.OptimizationProblem(optf, p_LBFGS_init)
res_lbfgs      = Optimization.solve(optprob_LBFGS, LBFGS(linesearch = BackTracking()); callback = callback, maxiters = 50000)
lbfgs_loss     = loss_history[end]
println("LBFGS Final Loss: ", lbfgs_loss)

p_final    = ComponentArray(res_lbfgs.u, pax)
sol_final  = solve(prob_pred, Tsit5(); p = p_final, saveat = tdata)
plot_u0_fitting_periodic(sol_final .* max_vals, tdata, ode_data .* max_vals, u0, iter)

final_loss_plot = plot(loss_history, xlabel = "Iteration", ylabel = "Loss",
                       title = "Loss Curve κ-only (ADAM + LBFGS)", lw = 2, label = "Loss")
savefig(final_loss_plot, "PEM_KAPPA_ONLY/final_loss_curve.png")
display(final_loss_plot)
