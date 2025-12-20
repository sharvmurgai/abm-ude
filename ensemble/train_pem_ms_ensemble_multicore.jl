# Multi-core train_pem_ms_ensemble_multicoe.jl
#
# run each seed independently on a separate Julia *process* using pmap.
#
# usage:
#   julia --project=julia_test --threads=1 -p auto train_pem_ms_ensemble_multicore.jl
#   julia --project=julia_test --threads=1 -p 8 train_pem_ms_ensemble_multicore.jl

using Distributed

const SCRIPT_DIR = dirname(@__FILE__)
cd(SCRIPT_DIR)

if nprocs() == 1
    nenv = try
        parse(Int, get(ENV, "JULIA_NWORKERS", "0"))
    catch
        0
    end
    if nenv > 0
        addprocs(nenv)
    else
        @info "No worker processes detected. Start Julia with `-p N` (e.g. `-p auto`) or set JULIA_NWORKERS."
    end
end

using Dates
const SCRIPT_NAME   = splitext(basename(@__FILE__))[1]
const RUN_TIMESTAMP = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
const OUTPUT_ROOT   = "MS_PEM_" * SCRIPT_NAME * "_" * RUN_TIMESTAMP
mkpath(OUTPUT_ROOT)
println("Output root directory: $OUTPUT_ROOT")

@everywhere const SCRIPT_DIR = $SCRIPT_DIR
@everywhere const OUTPUT_ROOT = $OUTPUT_ROOT
@everywhere cd(SCRIPT_DIR)

@everywhere begin
    # Project + packages
    using Pkg
    #Pkg.activate("julia_test")

    if !haskey(ENV, "GKSwstype")
        ENV["GKSwstype"] = "100"
    end

    using MAT, JLD2, Plots, Measures, Random, LaTeXStrings
    using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq
    using LinearAlgebra, DiffEqFlux, Optim
    using SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
    using ComponentArrays, Lux, Zygote, StableRNGs, Statistics
    using CSV, DataFrames
    using ReverseDiff
    using LineSearches
    using DiffEqFlux: group_ranges
    using OptimizationPolyalgorithms
    using Dates
    using Interpolations
    using Colors

    # Avoid BLAS oversubscription when using many processes
    LinearAlgebra.BLAS.set_num_threads(1)

    # We now treat "MS+PEM" as a single fused ODE
    const TRAIN_MODE = :fused   

    # Ensemble settings
    const N_SEEDS    = 100
    const FIRST_SEED = 0      # TBD: for now, fixed seeds 0 to 99 

    # Smoke test vs full run
    const SMOKE_TEST        = false        # true
    const SMOKE_ADAM_ITERS  = 10
    const SMOKE_LBFGS_ITERS = 2         # if 0, skip LBFGS in smoke

    # "Full" iteration counts (per seed)
    const ADAM_FULL_ITERS   = 200     # adjust as needed
    const LBFGS_FULL_ITERS  = 50      # adjust as needed

    ADAM_maxiters  = SMOKE_TEST ? SMOKE_ADAM_ITERS  : ADAM_FULL_ITERS
    LBFGS_maxiters = SMOKE_TEST ? SMOKE_LBFGS_ITERS : LBFGS_FULL_ITERS

    const PLOT_EVERY = 100000  # plot every N optimizer steps
 
    # Strength of the PEM-like assimilation term inside the fused ODE
    const GAMMA_PEM = 1.0  # TBD: tune this (0.1, 0.5, 1.0, 2.0)

    global OUTPUT_DIR = OUTPUT_ROOT

    model_architecture = "_1NN"
    utils_adjust = ""
    method = model_architecture * utils_adjust

    utils_path = "utils" * method * ".jl"   # "utils_1NN.jl"
    include(utils_path)                     

    model     = "SEInsIsIaDR"
    est_all   = false
    scale_NN  = true
    num_inputs = [6]

    # Data file
    const DATA_FILE = "avg_output.dat"  
    df = DataFrame(CSV.File(DATA_FILE, delim=' ', ignorerepeated=true))
    println("[pid=$(myid())] Loaded data from $DATA_FILE with columns: ", names(df))

    selected_df, tdata, tspan, train_n_days, ode_data, u0, max_vals =
        process_data(df, model)

    S, T = size(ode_data)
    @assert S == 6

    const STATE_NAMES = ["Never","Exposed","Presymptomatic",
                         "Symptomatic","Asymptomatic","Deaths"]

    datasize   = length(tdata)
    group_size = 5
    ranges = group_ranges(datasize, group_size)

    # Interpolators for MS nudging and PEM-like term
    const data_interpolators = [
        Interpolations.LinearInterpolation(tdata, ode_data[i, :]) for i in 1:S
    ]

    function make_linear_interpolator(tgrid::AbstractVector, Y::AbstractMatrix)
        @assert size(Y, 2) == length(tgrid)
        cols = [@view Y[:, i] for i in 1:length(tgrid)]
        function y_of_t(t::Real)
            if t <= tgrid[1]
                return copy(cols[1])
            elseif t >= tgrid[end]
                return copy(cols[end])
            end
            i  = searchsortedlast(tgrid, t)
            i  = clamp(i, 1, length(tgrid) - 1)
            t0 = tgrid[i]; t1 = tgrid[i+1]
            w  = (t - t0) / (t1 - t0)
            @. (1 - w) * cols[i] + w * cols[i+1]
        end
        return y_of_t
    end

    const y_of_t = make_linear_interpolator(tdata, ode_data)

    # Fixed Ki nudging gain (original MS nudging)
    const FIXED_Ki = fill(10.0, S)

    # κ-NETWORK + PARAMS
    softplus(x) = log1p(exp(x))

    nn_kappa = Chain(
        Dense(6, 32, tanh),
        Dense(32, 16, tanh),
        Dense(16, 1)  # softplus outside
    )

    # Template setup (for axes / shapes)
    p_init_template, st_kappa_template = Lux.setup(Xoshiro(0), nn_kappa)
    p_ode_template = ComponentArray(fa = 0.5, fr = 0.5, scale_Kappa = 1.0)
    p0_template    = ComponentArray(; θ = p_init_template, ode_pars = p_ode_template)

    ps_template = p0_template
    pd_template, pax = getdata(ps_template), getaxes(ps_template)

    # Dense layers are stateless; we can reuse one state object
    const ST_KAPPA = st_kappa_template

    # Fixed epidemiological parameters
    const p_true = (T_E = 5.0, T_In = 7.0, T_i = 10.0,
                    eta_a = 0.5, p_trans = 3.7/100.0)

    """
        kappa_from_state(u, p)

    Compute κ(u; θ) > 0 using the shared nn_kappa and softplus,
    optionally scaled by scale_Kappa.
    """
    function kappa_from_state(u::AbstractVector, p)
        S0, E, Ins, Is, Ia, D = u
        N0 = S0 + E + Ins + Is + Ia
        if N0 < 1.0
            N0 = 1.0
        end

        u_norm = Float32[S0/N0, E/N0, Ins/N0, Is/N0, Ia/N0, D/N0]
        out, _ = nn_kappa(u_norm, p.θ, ST_KAPPA)
        κ_raw = out[1]
        κ = softplus(κ_raw) + 1e-6   

        if scale_NN
            κ *= abs(p.ode_pars.scale_Kappa)
        end
        return κ
    end

    # MECHANISTIC SEInsIsIaDR RHS
    function SEInsIsIaDR_6_state!(du, u, p, t; p_true = p_true)
        # Unpack fixed epidemiological params
        T_E, T_In, T_i, eta_a, p_trans = p_true
        # Estimated parameters (always positive)
        fa = abs(p.ode_pars.fa)
        fr = abs(p.ode_pars.fr)

        S0, E, Ins, Is, Ia, D = u
        κ = kappa_from_state(u, p)

        T_ins = T_In - T_E
        T_s   = T_E + T_i - T_In

        dem = S0 + E + Ins + Is + Ia
        if dem < 1.0
            dem = 1.0
        end
        num = eta_a * Ins + Is + eta_a * Ia
        λ = p_trans * κ * num / dem

        du[1] = -λ * S0
        du[2] = λ * S0 - E / T_E
        du[3] = (1.0 - fa) * E / T_E - Ins / T_ins
        du[4] = Ins / T_ins - Is / T_s
        du[5] = fa * E / T_E - Ia / T_i
        du[6] = (1.0 - fr) * Is / T_s
    end

    # FUSED MS+PEM ODE (single dynamical system)
    """
        fused_ms_pem!(du, u, p, t)

    Single ODE that combines:
      - mechanistic SEInsIsIaDR dynamics,
      - original nudging (Ki * (data - u)),
      - PEM-like assimilation term GAMMA_PEM * κ(u;θ) * (y(t) - u).

    This is the only ODE used for multiple shooting and plotting.
    """
    function fused_ms_pem!(du, u, p, t)
        # Start with mechanistic SEInsIsIaDR derivative
        SEInsIsIaDR_6_state!(du, u, p, t)

        # Original nudging toward data (Ki * (data - u))
        y_data = [interp(t) for interp in data_interpolators]
        du .+= FIXED_Ki .* (y_data .- u)

        # PEM-like assimilation term using κ(u;θ) and y_of_t(t)
        κ = kappa_from_state(u, p)
        y_vec = y_of_t(t)
        du .+= GAMMA_PEM * κ .* (y_vec .- u)
    end

    # Fused ODE problem used everywhere (multi-shoot + final solves)
    prob_fused_template = ODEProblem(fused_ms_pem!, u0, tspan, p0_template, saveat = tdata)

    # LOSS: MULTIPLE SHOOTING ON FUSED ODE
    # Normalized-per-state error (same idea as your original MS loss)
    function loss_function_ms(data, pred)
        scales = mean(abs.(data), dims = 2) .+ 1e-6
        scaled_error = (data .- pred) ./ scales
        return sum(abs2, scaled_error)
    end

    continuity_term = 100.0

    function loss_fused(pvec)
        p_struct = ComponentArray(pvec, pax)
        loss, _ = multiple_shoot(p_struct, ode_data, tdata,
                                 prob_fused_template, loss_function_ms,
                                 Tsit5(), group_size;
                                 continuity_term = continuity_term)
        return loss
    end

    # Single objective: fused MS+PEM ODE loss
    function loss_total(pvec)
        return loss_fused(pvec)
    end

    function plot_fit_joint(sol_arr, tdata, ode_data, tag;
                            state_names = STATE_NAMES,
                            outdir = OUTPUT_DIR)
        mkpath(outdir)
        S, T = size(sol_arr)

        for i in 1:S
            state = state_names[i]

            plt = plot(
                tdata, ode_data[i, :],
                label = "Observed",
                xlabel = "Time",
                ylabel = "Population",
                title  = "$state ($tag)",
            )
            plot!(plt, tdata, sol_arr[i, :],
                  label = "MS+PEM (fused UDE)", linestyle = :solid)

            fname = joinpath(outdir,
                             "fit_$(replace(state, ' '=>'_'))_joint_$(tag).png")
            println("  [pid=$(myid()) plot_fit_joint] Saving: $fname")
            savefig(plt, fname)
        end
    end

    # OPTIMIZATION FUNCTION 
    # and CALLBACK
    adtype = Optimization.AutoZygote()
    optf   = Optimization.OptimizationFunction((x, p) -> loss_total(x), adtype)

    loss_history = Float64[]
    iter = 0

    callback = function (state, l; doplot = true)
        global iter, loss_history
        iter += 1
        push!(loss_history, l)

        if iter % 10 == 0
            println("[pid=$(myid())] callback: iter=$iter, loss=$l")
        end

        if doplot && iter % PLOT_EVERY == 0
            println("[pid=$(myid())] >>> PLOTTING at iter=$iter, loss=$l")
            p_current = ComponentArray(state.u, pax)

            # For the fused model, we only solve the fused ODE
            sol_fused = solve(prob_fused_template, Tsit5(); p = p_current, saveat = tdata)
            if SciMLBase.successful_retcode(sol_fused)
                sol_den  = Array(sol_fused) .* max_vals
                data_den = ode_data        .* max_vals
                tag = "iter_$(lpad(iter, 6, '0'))"
                plot_fit_joint(sol_den, tdata, data_den, tag; outdir = OUTPUT_DIR)
            else
                println("fused solve failed at iter=$iter, retcode=$(sol_fused.retcode)")
            end
        end

        return false
    end

    # SINGLE-SEED RUNNER (used by pmap)
    function run_seed(run_idx::Int)
        # 1. Deterministic Seed Calculation
        rng_seed = FIRST_SEED + run_idx - 1

        # 2. SET GLOBAL SEED (Strict Reproducibility)
        Random.seed!(rng_seed)

        println("\n[pid=$(myid())] Ensemble run $(run_idx)/$(N_SEEDS), RNG seed = $(rng_seed) ")

        # Per-seed output directory (unique per seed)
        seed_dir = joinpath(OUTPUT_ROOT, "seed_$(lpad(run_idx, 3, '0'))")
        mkpath(seed_dir)

        # Ensure callback writes into the correct directory (per worker)
        global OUTPUT_DIR = seed_dir

        # Reset loss history & iter counter (per seed)
        global loss_history = Float64[]
        global iter = 0

        try
            # SET LOCAL SEED FOR LUX (deterministic per seed)
            p_init_nn_seed, _ = Lux.setup(Xoshiro(rng_seed), nn_kappa)
            p_ode_seed    = ComponentArray(fa = 0.5, fr = 0.5, scale_Kappa = 1.0)
            p_struct_seed = ComponentArray(; θ = p_init_nn_seed, ode_pars = p_ode_seed)

            # Flatten to vector for optimizer 
            pd_seed, _ = getdata(ComponentArray(p_struct_seed, pax)), pax

            # Optimization problem for this seed
            optprob_seed = Optimization.OptimizationProblem(optf, pd_seed)

            println("[pid=$(myid())] === ADAM phase ($TRAIN_MODE) for seed $(run_idx) ===")
            res_adam = Optimization.solve(
                optprob_seed,
                ADAM(0.001);
                callback = callback,
                maxiters = ADAM_maxiters,
            )

            # LBFGS refinement
            p_final_vec=res_adam.u
            if LBFGS_maxiters > 0
                p_LBFGS_init = getdata(ComponentArray(res_adam.u, pax))
                loss_history = Float64[]  # reset for LBFGS history only
                optprob_LBFGS = Optimization.OptimizationProblem(optf, p_LBFGS_init)

                println("[pid=$(myid())] === LBFGS phase ($TRAIN_MODE) for seed $(run_idx) ===")
                res_lbfgs = Optimization.solve(
                    optprob_LBFGS,
                    LBFGS(linesearch = BackTracking());
                    callback = callback,
                    maxiters = LBFGS_maxiters,
                )
                p_final_vec = res_lbfgs.u
            else
                println("[pid=$(myid())] Skipping LBFGS for seed $(run_idx)")
                p_final_vec = res_adam.u
            end

            p_final = ComponentArray(p_final_vec, pax)

            # Final fused solve & plot for this seed
            sol_fused_final = solve(prob_fused_template, Tsit5(); p = p_final, saveat = tdata)
            if SciMLBase.successful_retcode(sol_fused_final)
                sol_den_final  = Array(sol_fused_final) .* max_vals
                data_den_final = ode_data              .* max_vals

                # Metrics Calculation 
                errors = sol_den_final .- data_den_final
                rmse_states = [sqrt(mean(errors[i, :].^2)) for i in 1:S]
                total_rmse = mean(rmse_states)

                # Final loss for this seed
                final_loss = isempty(loss_history) ? loss_total(p_final_vec) : loss_history[end]

                df_metrics = DataFrame(
                    seed = run_idx,
                    rng_seed = rng_seed,
                    final_loss = final_loss,
                    total_rmse = total_rmse,
                    rmse_never = rmse_states[1],
                    rmse_exposed = rmse_states[2],
                    rmse_presymptomatic = rmse_states[3],
                    rmse_symptomatic = rmse_states[4],
                    rmse_asymptomatic = rmse_states[5],
                    rmse_deaths = rmse_states[6],
                )

                CSV.write(joinpath(OUTPUT_DIR,
                         "metrics_seed_$(lpad(run_idx, 3, '0')).csv"), df_metrics)
                println("[pid=$(myid())]   [metrics] Saved metrics_seed_$(lpad(run_idx, 3, '0')).csv")

                plot_fit_joint(sol_den_final, tdata, data_den_final,
                               "final_seed_$(lpad(run_idx, 3, '0'))"; outdir = OUTPUT_DIR)


                ## save to CSV
                df_ts = DataFrame(time = tdata)
                df_ts.seed = fill(run_idx, length(tdata))
                df_ts.rng_seed = fill(rng_seed, length(tdata))

                for (i, nm) in enumerate(STATE_NAMES)
                    key = lowercase(replace(nm, " " => "_"))
                    df_ts[!, "obs_" * key]  = vec(data_den_final[i, :])
                    df_ts[!, "pred_" * key] = vec(sol_den_final[i, :])
                end

                CSV.write(joinpath(OUTPUT_DIR,
                        "timeseries_seed_$(lpad(run_idx, 3, '0')).csv"), df_ts)
                println("[pid=$(myid())]   [csv] Saved timeseries_seed_$(lpad(run_idx, 3, '0')).csv")

                ##

            else
                println("[pid=$(myid())]  [warn] final fused solve failed for seed $(run_idx), retcode=$(sol_fused_final.retcode)")
            end

            # Loss curve for this seed
            plt_loss = plot(
                loss_history,
                xlabel = "Iteration",
                ylabel = "Loss",
                title  = "Loss curve ($TRAIN_MODE, seed=$(run_idx))",
                lw     = 2,
                label  = "loss",
            )
            savefig(joinpath(OUTPUT_DIR,
                             "loss_curve_seed_$(lpad(run_idx, 3, '0'))_$TRAIN_MODE.png"))
            
            CSV.write(joinpath(OUTPUT_DIR,
                    "loss_history_seed_$(lpad(run_idx, 3, '0')).csv"),
                    DataFrame(iter = 1:length(loss_history), loss = loss_history))
                    
            return (seed = run_idx, rng_seed = rng_seed, status = "ok", pid = myid())
        catch err
            # Log error and continue other seeds
            errfile = joinpath(seed_dir, "error.txt")
            open(errfile, "w") do io
                println(io, "Seed: ", run_idx)
                println(io, "rng_seed: ", rng_seed)
                println(io, "pid: ", myid())
                println(io, "\n--- ERROR ---")
                showerror(io, err, catch_backtrace())
            end
            println("[pid=$(myid())] ERROR in seed $(run_idx). Logged to $errfile")
            return (seed = run_idx, rng_seed = rng_seed, status = "error", pid = myid())
        end
    end
end # @everywhere begin

# Distribute seeds + aggregate metrics
if myid() == 1
    println("\n========== PARALLEL ENSEMBLE ==========")
    println("nprocs() = $(nprocs()), workers() = $(workers())")

    seed_indices = collect(1:N_SEEDS)

    results = if nworkers() == 0
        @info "No workers available; running ensemble serially on pid=1."
        map(run_seed, seed_indices)
    else
        pmap(run_seed, seed_indices)
    end

    using DataFrames, CSV
    df_status = DataFrame(results)
    CSV.write(joinpath(OUTPUT_ROOT, "seed_status.csv"), df_status)
    println("Saved per-seed status to: ", joinpath(OUTPUT_ROOT, "seed_status.csv"))

    # post-ensemble analysis: aggregate metrics
    println("\n AGGREGATING RESULTS ")

    metric_files = String[]
    for (root, dirs, files) in walkdir(OUTPUT_ROOT)
        for file in files
            if startswith(file, "metrics_seed_") && endswith(file, ".csv")
                push!(metric_files, joinpath(root, file))
            end
        end
    end

    if isempty(metric_files)
        println("No metric files found to aggregate!")
    else
        df_all = vcat([DataFrame(CSV.File(f)) for f in metric_files]...)
        sort!(df_all, :seed)

        mean_rmse = mean(df_all.total_rmse)
        std_rmse  = std(df_all.total_rmse)

        mean_loss = mean(df_all.final_loss)
        std_loss  = std(df_all.final_loss)

        println("-"^40)
        println("RESULTS FOR TRAIN_MODE = $TRAIN_MODE (N=$(nrow(df_all)))")
        println("-"^40)
        println("Total RMSE: $(round(mean_rmse, digits=4)) ± $(round(std_rmse, digits=4))")
        println("Final Loss: $(round(mean_loss, digits=4)) ± $(round(std_loss, digits=4))")
        println("-"^40)

        summary_path = joinpath(OUTPUT_ROOT, "ensemble_summary.txt")
        open(summary_path, "w") do io
            println(io, "TRAIN_MODE: $TRAIN_MODE")
            println(io, "N_SEEDS: $(nrow(df_all))")
            println(io, "GAMMA_PEM: $GAMMA_PEM")
            println(io, "Mean RMSE: $mean_rmse")
            println(io, "Std RMSE:  $std_rmse")
            println(io, "Mean Loss: $mean_loss")
            println(io, "Std Loss:  $std_loss")
            println(io, "\n--- FULL TABLE ---")
            println(io, df_all)
        end

        CSV.write(joinpath(OUTPUT_ROOT, "ensemble_all_seeds.csv"), df_all)
        println("Saved summary to $summary_path")
        println("Saved all-seeds CSV to $(joinpath(OUTPUT_ROOT, "ensemble_all_seeds.csv"))")
    end
end



