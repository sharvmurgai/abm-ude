cd(dirname(@__FILE__))

using Pkg
Pkg.activate("julia_test")

using MAT, JLD2
using CSV, DataFrames
using Random, StableRNGs, Statistics, LinearAlgebra
BLAS.set_num_threads(1)
using Colors
using Dates
using OrdinaryDiffEq, ModelingToolkit
using SciMLSensitivity, SciMLBase
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays
using Lux, Zygote, StaticArrays
using Plots, Measures
using LineSearches
using Base.Threads   # Emsemble changes

include("utils_1NN.jl")   # provides: process_data(df, model)
const MODEL_NAME     = "SEInsIsIaDR"
const ADAM_maxiters  = 150_000
const LBFGS_maxiters = 15_000
const N_RUNS         = 100
#const ADAM_maxiters  = 15
#const LBFGS_maxiters = 5
#const N_RUNS         = 5

# Base name for runs + timestamped OUTROOT
const OUTROOT_BASE = "runs_nov16"
const OUTROOT = OUTROOT_BASE * "_" * Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

# Fixed epidemiological parameters; ONLY κ and K_obs are learned
const p_true = (T_E=5.0, T_In=7.0, T_i=10.0, eta_a=0.5, p_trans=3.7/100.0,
                fa=0.35, fr=0.90)

# States: 6 observed + 1 recovered (R) internally
const STATE_NAMES = ["Never","Exposed","Presymptomatic","Symptomatic","Asymptomatic","Deaths"]
const S = length(STATE_NAMES)

# Optional mask (1.0 = nudge on that state, 0.0 = off)
const OBS_MASK = ones(Float64, S)

df = DataFrame(CSV.File("avg_output.dat", delim=' ', ignorerepeated=true))
selected_df, tdata, tspan, train_n_days, data_norm, u0_norm, max_vals = process_data(df, MODEL_NAME)

# Denormalize back to original scale for mechanistic ODE fitting
data = data_norm .* max_vals            # 6×T
u0_6 = u0_norm   .* max_vals            # 6

# Recovered initial guess (from df if available; else 0)
R0_guess = hasproperty(df, :Immune) ? Float64(df[1, :Immune]) : 0.0
N0 = sum(u0_6) + R0_guess               # total incl. R, excl. D in denominator where needed
R0 = max(R0_guess, (N0 - u0_6[end]) - sum(u0_6[1:5]))  # ensure nonnegative
u0 = vcat(u0_6, R0)  # 7 states: S,E,Ins,Is,Ia,D,R

function make_linear_interpolator(tgrid::AbstractVector, Y::AbstractMatrix)
    cols = [@view Y[:, i] for i in 1:length(tgrid)]
    function y_of_t(t::Real)
        if t <= tgrid[1]; return copy(cols[1]); end
        if t >= tgrid[end]; return copy(cols[end]); end
        i = clamp(searchsortedlast(tgrid, t), 1, length(tgrid)-1)
        t0 = tgrid[i]; t1 = tgrid[i+1]; w = (t - t0) / (t1 - t0)
        @. (1 - w) * cols[i] + w * cols[i+1]
    end
    return y_of_t
end
const y_of_t = make_linear_interpolator(tdata, data)  # 6-vector

function save_predictions_csv(sol_arr::AbstractArray, tdata::AbstractVector, data_arr::AbstractArray;
                              outdir::AbstractString = "PEM_UDE_KAPPA_OG__csv", tag::AbstractString = "")
    mkpath(outdir)
    cols = Dict{Symbol, Any}()
    cols[:time] = collect(tdata)
    @inbounds for (i, nm) in enumerate(STATE_NAMES)
        cols[Symbol("$(nm)_obs")]  = vec(data_arr[i, :])
        cols[Symbol("$(nm)_pred")] = vec(sol_arr[i, :])
    end
    outpath = joinpath(outdir, isempty(tag) ? "predictions.csv" : "predictions_$(tag).csv")
    CSV.write(outpath, DataFrame(cols))
end

# Simple R² on flattened observed states
function r2_score(y_true::AbstractArray, y_pred::AbstractArray)
    μ = mean(y_true)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- μ).^2)
    return ss_tot ≈ 0 ? NaN : 1 - (ss_res / ss_tot)
end

# Metric container
struct FitMetrics
    mse_overall::Float64
    mae_overall::Float64
    r2_overall::Float64
    mse_per_state::Vector{Float64}
    seed::Int
    adam_loss::Float64
    lbfgs_loss::Float64
    elapsed_s::Float64
    retcode::Any
end

softplus(x) = log1p(exp(x))

function build_run(seed::Int;
                   model::String=MODEL_NAME,
                   p_true::NamedTuple=p_true,
                   obs_mask::AbstractVector{<:Real}=OBS_MASK,
                   tdata::AbstractVector=tdata,
                   data::AbstractMatrix=data,
                   u0::AbstractVector=u0,
                   N0::Real=N0)

    rng = StableRNGs.StableRNG(seed)

    # κ(u;θ) network
    nn_kappa_local = Chain(
        Dense(6, 32, tanh),
        Dense(32, 16, tanh),
        Dense(16, 1)
    )
    pκ_init_local, stκ_local = Lux.setup(rng, nn_kappa_local)

    # Trainable per-state nudging gains for 6 observed states
    Kraw_init_local = fill(-2.0, S)  # softplus -> small positive

    # Pack parameters
    p0_local = ComponentArray(; θκ=pκ_init_local, Kraw=Kraw_init_local)

    # RHS (PEM nudging on first 6 states)
    function rhs!(du, u, p, t)
        T_E   = p_true.T_E    ; T_In  = p_true.T_In
        T_i   = p_true.T_i    ; eta_a = p_true.eta_a
        p_tr  = p_true.p_trans
        fa    = p_true.fa     ; fr    = p_true.fr

        S, E, Ins, Is, Ia, D, R = u

        # κ(u;θ)
        NminusD = max(S + E + Ins + Is + Ia + R, 1e-9)
        xκ = @SVector [E/NminusD, Ins/NminusD, Is/NminusD, Ia/NminusD, D/max(N0,1e-9), R/NminusD]
        κ = softplus(nn_kappa_local(xκ, p.θκ, stκ_local)[1][1]) + 1e-8

        # Force of infection
        λ = p_tr * κ * (eta_a*Ins + Is + eta_a*Ia) / max((S + E + Ins + Is + Ia + R), 1e-9)

        T_ins = T_In - T_E
        T_s   = T_E + T_i - T_In

        du[1] = -λ * S
        du[2] =  λ * S - E / T_E
        du[3] = (1.0 - fa) * E / T_E - Ins / T_ins
        du[4] =  Ins / T_ins - Is / T_s
        du[5] =  fa * E / T_E - Ia / T_i
        du[6] = (1.0 - fr) * Is / T_s
        du[7] =  fr * Is / T_s + Ia / T_i

        # PEM nudging on observed states 1..6
        y = y_of_t(t)  # 6-vector
        @inbounds for i in 1:6
            Ki = softplus(p.Kraw[i])             # ≥ 0
            du[i] += (obs_mask[i] * Ki) * (y[i] - u[i])
        end
        return nothing
    end

    prob_local = ODEProblem((du,u,p,t)->rhs!(du,u,p,t),
                            u0, (tdata[1], tdata[end]), p0_local; saveat=tdata)

    # Loss on observed states only (1..6)
    obs_idx = 1:6
    function loss_fn_local(pvec)
        p_struct = ComponentArray(pvec, getaxes(p0_local))
        sol = solve(remake(prob_local, p=p_struct), Tsit5(); saveat=tdata)
        if !SciMLBase.successful_retcode(sol.retcode); return Inf; end
        pred = Array(sol)[obs_idx, :]
        return mean(abs2.(data .- pred))
    end

    return (; prob_local, p0_local, loss_fn_local, nn_kappa_local, stκ_local)
end

function train_once(seed::Int; outroot::String=OUTROOT)
    run_dir = joinpath(outroot, "run_$(seed)")
    mkpath(run_dir)

    # Build per-seed run
    built = build_run(seed)
    prob_local = built.prob_local
    p0_local   = built.p0_local
    loss_fn_local = built.loss_fn_local

    adtype = Optimization.AutoZygote()
    optf   = Optimization.OptimizationFunction((x,_)->loss_fn_local(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, getdata(p0_local))

    # Loss history & callback (covers ADAM + LBFGS)
    loss_hist = Float64[]
    callback = function (p, l)
        push!(loss_hist, l)
        if length(loss_hist) % 1000 == 0
            println(l)
            print("\n")
        end
        return false
    end

    t0 = time()

    # ADAM
    res_adam = Optimization.solve(optprob, ADAM(1e-3); callback=callback, maxiters=ADAM_maxiters)
    adam_loss = res_adam.minimum

    # LBFGS
    p_LBFGS_init = getdata(ComponentArray(res_adam.u, getaxes(p0_local)))
    optprob_LBFGS = Optimization.OptimizationProblem(optf, p_LBFGS_init)
    res_lbfgs = Optimization.solve(optprob_LBFGS, LBFGS(linesearch=BackTracking()); callback=callback, maxiters=LBFGS_maxiters)
    lbfgs_loss = res_lbfgs.minimum
    elapsed = time() - t0

    # Save loss history (CSV only; plotting happens later)
    if !isempty(loss_hist)
        loss_df = DataFrame(iter = 1:length(loss_hist), loss = loss_hist)
        CSV.write(joinpath(run_dir, "loss_history.csv"), loss_df)
    end

    # Final solution & predictions
    p_final = ComponentArray(res_lbfgs.u, getaxes(p0_local))
    sol_final = solve(remake(prob_local, p=p_final), Tsit5(); saveat=tdata)
    pred_final = Array(sol_final)[1:6, :]

    # Save per-run predictions
    save_predictions_csv(pred_final, tdata, data; outdir=run_dir, tag="final")

    # Compute and save metrics
    y_true = data
    y_pred = pred_final
    mse_per_state = [mean((y_true[i, :] .- y_pred[i, :]).^2) for i in 1:size(y_true,1)]
    mae_overall = mean(abs.(y_true .- y_pred))
    mse_overall = mean((y_true .- y_pred).^2)
    r2_overall  = r2_score(vec(y_true), vec(y_pred))

    CSV.write(joinpath(run_dir, "metrics.csv"),
        DataFrame(; seed=seed,
                    adam_loss=adam_loss,
                    lbfgs_loss=lbfgs_loss,
                    mse_overall=mse_overall,
                    mae_overall=mae_overall,
                    r2_overall=r2_overall,
                    mse_Never=mse_per_state[1],
                    mse_Exposed=mse_per_state[2],
                    mse_Presymptomatic=mse_per_state[3],
                    mse_Symptomatic=mse_per_state[4],
                    mse_Asymptomatic=mse_per_state[5],
                    mse_Deaths=mse_per_state[6],
                    elapsed_s=elapsed,
                    retcode=string(sol_final.retcode)
        )
    )

    return FitMetrics(mse_overall, mae_overall, r2_overall, mse_per_state, seed,
                      adam_loss, lbfgs_loss, elapsed, sol_final.retcode)
end

function run_ensemble(; N::Int=N_RUNS, outroot::String=OUTROOT)
    mkpath(outroot)
    results = Vector{FitMetrics}(undef, N)

    @threads for i in 1:N
        seed = i
        try
            println(">>> Starting run $seed on thread $(threadid()) at $(Dates.now())")
            results[i] = train_once(seed; outroot=outroot)  # NO plotting in threads
        catch e
            @warn "Run $seed failed" error=e
            results[i] = FitMetrics(NaN, NaN, NaN, fill(NaN, 6), seed, NaN, NaN, NaN, :Failure)
        end
    end

    # Aggregate to summary table
    df_sum = DataFrame(
        seed = [r.seed for r in results],
        adam_loss = [r.adam_loss for r in results],
        lbfgs_loss = [r.lbfgs_loss for r in results],
        mse_overall = [r.mse_overall for r in results],
        mae_overall = [r.mae_overall for r in results],
        r2_overall  = [r.r2_overall for r in results],
        mse_Never   = [r.mse_per_state[1] for r in results],
        mse_Exposed = [r.mse_per_state[2] for r in results],
        mse_Presymptomatic = [r.mse_per_state[3] for r in results],
        mse_Symptomatic    = [r.mse_per_state[4] for r in results],
        mse_Asymptomatic   = [r.mse_per_state[5] for r in results],
        mse_Deaths         = [r.mse_per_state[6] for r in results],
        elapsed_s   = [r.elapsed_s for r in results],
        retcode     = [string(r.retcode) for r in results]
    )

    CSV.write(joinpath(outroot, "ensemble_summary.csv"), df_sum)

    # Robust summary over successful runs
    good_df = filter(:mse_overall => isfinite, df_sum)
    Ngood   = nrow(good_df)

    if Ngood == 0
        println("=== Ensemble: no successful runs ===")
        return df_sum
    end

    μ_mse = mean(good_df.mse_overall)
    σ_mse = std(good_df.mse_overall)
    μ_r2  = mean(good_df.r2_overall)
    σ_r2  = std(good_df.r2_overall)

    println("=== Ensemble (N=$Ngood / $N) ===")
    println("Overall MSE: $(round(μ_mse, sigdigits=5)) ± $(round(σ_mse, sigdigits=5))")
    println("Overall R² : $(round(μ_r2,  sigdigits=5)) ± $(round(σ_r2,  sigdigits=5))")

    return df_sum
end

function generate_plots(; outroot::String=OUTROOT, N::Int=N_RUNS)
    println("[INFO] Generating plots from CSVs in $outroot")

    # Per-seed plots
    for seed in 1:N
        run_dir = joinpath(outroot, "run_$(seed)")

        # 1) Obs vs pred per state
        pred_path = joinpath(run_dir, "predictions_final.csv")
        if isfile(pred_path)
            df = CSV.read(pred_path, DataFrame)
            names_df = names(df)
            for state in STATE_NAMES
                obs_col  = Symbol("$(state)_obs")
                pred_col = Symbol("$(state)_pred")
                if !(obs_col in names_df && pred_col in names_df)
                    @warn "Missing columns for state $state in seed $seed" obs_col pred_col
                    continue
                end
                plt = plot(df.time, df[!, obs_col],
                           label  = "obs",
                           xlabel = "time",
                           ylabel = "count",
                           title  = "Seed $seed – $state")
                plot!(plt, df.time, df[!, pred_col], label="pred")
                savefig(plt, joinpath(run_dir, "state_$(state)_obs_vs_pred.png"))
            end
        end

        # 2) Loss curve per seed (if loss_history.csv exists)
        loss_path = joinpath(run_dir, "loss_history.csv")
        if isfile(loss_path)
            ldf = CSV.read(loss_path, DataFrame)
            cols = names(ldf)
            has_iter = (:iter in cols) || ("iter" in cols)
            has_loss = (:loss in cols) || ("loss" in cols)
            if has_iter && has_loss
                iter_col = :iter in cols ? :iter : Symbol("iter")
                loss_col = :loss in cols ? :loss : Symbol("loss")
                plt_loss = plot(ldf[!, iter_col], ldf[!, loss_col],
                                xlabel = "Iteration",
                                ylabel = "Loss",
                                yscale = :log10,
                                title  = "Loss curve (seed = $seed)")
                savefig(plt_loss, joinpath(run_dir, "loss_curve.png"))
            end
        end
    end

    # 3) Across-seed "loss per state" from ensemble_summary.csv
    sum_path = joinpath(outroot, "ensemble_summary.csv")
    if isfile(sum_path)
        df_sum = CSV.read(sum_path, DataFrame)
        cols_sum = names(df_sum)

        if (:seed in cols_sum) && (:mse_overall in cols_sum)
            plt_overall = plot(df_sum[!, :seed], df_sum[!, :mse_overall],
                               seriestype = :scatter,
                               xlabel = "seed",
                               ylabel = "MSE (overall)",
                               title  = "Overall MSE across seeds")
            savefig(plt_overall, joinpath(outroot, "overall_mse_across_seeds.png"))
        end

        for state in STATE_NAMES
            col = Symbol("mse_$(state)")
            if col in cols_sum
                plt_state = plot(df_sum[!, :seed], df_sum[!, col],
                                 seriestype = :scatter,
                                 xlabel = "seed",
                                 ylabel = "MSE",
                                 title  = "MSE for $state across seeds")
                savefig(plt_state, joinpath(outroot, "mse_$(state)_across_seeds.png"))
            end
        end
    end
end

function main()
    println("[INFO] OUTROOT = ", OUTROOT)
    println("[INFO] Starting PEM-UDE κ ensemble ($(N_RUNS) runs) @ ", Dates.now())
    df_sum = run_ensemble(; N=N_RUNS, outroot=OUTROOT)
    println("[INFO] Done @ ", Dates.now(), " → ", joinpath(OUTROOT, "ensemble_summary.csv"))

    # Now safely generate plots (sequential)
    generate_plots(; outroot=OUTROOT, N=N_RUNS)
end

main()
