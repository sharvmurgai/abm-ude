
cd(dirname(@__FILE__))
using Pkg
Pkg.activate("julia_test")

using MAT, JLD2, Plots, Measures, Random, LaTeXStrings
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq
using LinearAlgebra, DiffEqFlux, Optim #DiffEqSensitivity
using SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Zygote, StableRNGs, Statistics
using CSV, DataFrames
using ReverseDiff
using LineSearches
using DiffEqFlux: group_ranges
using OptimizationPolyalgorithms
using OptimizationOptimisers
using Colors
using Plots

using Dates
using Interpolations

const SCRIPT_NAME = splitext(basename(@__FILE__))[1]
const RUN_TIMESTAMP = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
const OUTPUT_DIR = "abm_ude_ms_$(SCRIPT_NAME)_$(RUN_TIMESTAMP)"

mkpath(OUTPUT_DIR)
println("Output will be saved to: $(OUTPUT_DIR)")

#Model and Paths Setup
model_architecture = "_1NN"
utils_adjust = ""
method = model_architecture * utils_adjust

utils_path = "utils" * method * ".jl"
models_path = "ODE_models" * model_architecture * ".jl"

ADAM_maxiters = 5000
# If this is not 5000, change Weights_dir_name to reflect that change
Weights_dir_name = "Weights" * method * "_200/"

include(utils_path)
# include(models_path) 
if !isdir(Weights_dir_name) mkdir(Weights_dir_name) end

model = "SEInsIsIaDR"
est_all = false
scale_NN = true

# TBD These values are taken from pemms_v8.jl file
const p_true = (T_E=5.0, T_In=7.0, T_i=10.0, eta_a=0.5, p_trans=3.7/100.0)

num_inputs = [6]  # (number of states, number of neural networks)

#Load Data
df = DataFrame(CSV.File("avg_output (1).dat", delim=' ', ignorerepeated=true))

# Process the data 
selected_df, tdata, tspan, train_n_days, ode_data, u0, max_vals = process_data(df, model)

const data_interpolators = [Interpolations.LinearInterpolation(tdata, ode_data[i, :]) for i in 1:6]

# Ki is a *fixed* hyperparameter, not a trainable parameter
# (10.0 is a good starting guess)
const FIXED_Ki = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

# Define group size for multiple shooting
datasize = length(tdata)  # Ensures datasize is correct
group_size = 5
ranges = group_ranges(datasize, group_size)
@show ranges
println("Total computed group ranges: ", length(ranges))


function SEInsIsIaDR_6_state!(du, u, p, t, p_true;
                            scale_NN::Bool=false, st_nn,
                            nudge=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Fixed parameters
    if isnothing(p_true)
        # Estimated parameters
        if scale_NN
            T_E, T_In, T_i, eta_a, p_trans, fa, fr, scale_Kappa = abs.(p.ode_pars)
        else

   T_E, T_In, T_i, eta_a, p_trans, fa, fr = abs.(p.ode_pars)
        end
    else
        if scale_NN

            # Fixed parameters
            T_E, T_In, T_i, eta_a, p_trans = p_true
            # Estimated parameters
            fa, fr, scale_Kappa = abs.(p.ode_pars)

  else
            # Fixed parameters
            T_E, T_In, T_i, eta_a, p_trans = p_true

            # Estimated parameters
            fa, fr = abs(p.ode_pars)
        end
    end

    S, E, Ins, Is, Ia, D = u

    pNN = p.pNN

    if !(t isa Float64)
        t = ReverseDiff.value(t)
    end

    N0 = S + E + Ins + Is + Ia
    if N0 < 1.0
        N0 = 1.0 # Avoid division by zero
    end

    output_NN, _ = nn([S/N0, E/N0, Ins/N0, Is/N0, Ia/N0, D/N0], pNN, st_nn)

    Kappa   = abs(output_NN[1])

    if scale_NN
        Kappa = Kappa * scale_Kappa
    end

    num = eta_a * Ins + Is + eta_a * Ia
    dem = S + E + Ins + Is + Ia # Interacting population
    if dem < 1.0
        dem = 1.0 # Avoid division by zero
    end
    lambda = p_trans * Kappa * num / dem


  T_ins = T_In - T_E
    T_s = T_E + T_i - T_In

    du[1] = dS = -lambda * S
    du[2] = dE = lambda * S - E / T_E
    du[3] = dIns = (1.0 - fa) * E / T_E - Ins / T_ins
    du[4] = dIs = Ins / T_ins - Is / T_s
    du[5] = dIa = fa * E / T_E - Ia / T_i
    du[6] = dD = (1.0 - fr) * Is / T_s

    du .+= nudge
end

# Define Neural ODE
nn = Chain(Dense(6, 16, tanh), Dense(16, 1))  # 6 input states → 1 output (Kappa)
p_nn, st = Lux.setup(Xoshiro(0), nn)


# We must also define the ODE parameters that SEInsIsIaDR! expects.
# Since `est_all` is false , `p_true` is used (set to nothing).
# We only need to estimate `fa`, `fr`, and `scale_Kappa`.
p_ode = ComponentArray(fa=0.5, fr=0.5, scale_Kappa=1.0) # Initial guesses

# Add all trainable parameters to p_init
p_init = ComponentArray(pNN = p_nn, ode_pars = p_ode)

function model_wrapper!(du, u, p, t)

    y_data = [interp(t) for interp in data_interpolators]
    nudging_term = FIXED_Ki .* (y_data - u)

    SEInsIsIaDR_6_state!(du, u, p, t, p_true; scale_NN=scale_NN, st_nn=st,
                         nudge=nudging_term)
end


prob_node = ODEProblem(model_wrapper!, u0, tspan, p_init, saveat = tdata)

# Loss Function
function loss_function_org(data, pred)
 return sum(abs2, data - pred)
end

# Loss Function
function loss_function(data, pred)
    # Calculate a scaling factor for each state (row).
    # We use the mean of the absolute value of the data for that state.
    # We add 1e-6 to avoid dividing by zero for any states that might be all-zero.
    scales = mean(abs.(data), dims=2) .+ 1e-6

    # Calculate the squared error, but first divide each state's error
    # by its own scale. This creates a "relative" or "normalized" error.
    scaled_error = (data .- pred) ./ scales

    # Return the sum of the squared relative errors.
    return sum(abs2, scaled_error)
end

continuity_term = 100
#continuity_term = 10  
#continuity_term = 200 

# Loss function for multiple shooting
function loss_multiple_shooting(p)
    ps = ComponentArray(p, pax)
    loss, _ = multiple_shoot(ps, ode_data, tdata, prob_node, loss_function, Tsit5(), group_size;
continuity_term)
    return loss # Return only the scalar loss
end

# Save predictions + ground truth to CSV 
const STATE_NAMES = ["Never","Exposed","Presymptomatic","Symptomatic","Asymptomatic","Deaths"]

function save_predictions_csv(sol, tdata, ode_data; max_vals=nothing, path="$(OUTPUT_DIR)/predictions.csv", tag="")
    mkpath(dirname(path))

    # Build (T × S) matrices for predicted and observed
    # sol[i, :] returns a Vector over time for state i
    S = length(STATE_NAMES)
    T = length(tdata)

    pred_mat = Array{Float64}(undef, T, S)
    for i in 1:S
        pred_mat[:, i] = Array(sol[i, :])

   end

    # ode_data is S × T in your code; transpose to T × S
    obs_mat = permutedims(ode_data, (2, 1))

    # Optional de-normalization
    if max_vals !== nothing
        @assert length(max_vals) == S "max_vals must have one scale per state."
        for i in 1:S
            pred_mat[:, i] .*= max_vals[i]
            obs_mat[:, i]  .*= max_vals[i]
        end
    end

    # Build a tidy DataFrame: time + pairs of (obs, pred) for each state
    cols = Dict{Symbol, Any}()
    cols[:time] = tdata
    for (i, name) in enumerate(STATE_NAMES)
        cols[Symbol("$(name)_obs")]  = obs_mat[:, i]


        cols[Symbol("$(name)_pred")] = pred_mat[:, i]
    end

    df_out = DataFrame(cols)

    # If tag provided, insert before extension
    if !isempty(tag)
        base, ext = splitext(path)
        path = base * "_" * tag * ext
    end

    CSV.write(path, df_out)
    println("Saved predictions to: $path")
end

# Optimization 
ps = ComponentArray(p_init)
pd, pax = getdata(ps), getaxes(ps)

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pd)
anim = Plots.Animation()
iter = 0  

using Plots

function plot_u0_fitting(res_ms, tdata, ode_data, u0)
    if res_ms === nothing
        println("WARNING: `res_ms` is undefined. Skipping u0 fitting plot.")
        return
    end

    println("Plotting u0 fitting...")
    plt = plot(title="Fitting of Initial Conditions u₀", xlabel="Time", ylabel="State Variables", legend=:topleft)

    # Convert res_ms.u back to structured parameters
    p_opt = ComponentArray(res_ms.u, pax)

    # Solve the ODE using the optimized parameters
    sol = solve(prob_node, Tsit5(), p=p_opt, saveat=tdata)

    # Plot the real data
    for i in 1:length(u0)
        scatter!(plt, tdata, ode_data[i, :], label="Observed - State $i", markershape=:circle)
    end

    # Plot the Neural ODE prediction
    for i in 1:length(u0)
        plot!(plt, tdata, sol[i, :], label="Predicted - State $i", linestyle=:dash)
    end

    savefig(plt, "$(OUTPUT_DIR)/u0_fitting.png")  # Save the figure
    display(plt)
end

function plot_u0_fitting_periodic(sol, tdata, ode_data, u0, iter; max_vals=nothing, lockdown_time=nothing)
mkpath("$(OUTPUT_DIR)")  #check if already exists?

    # Choose specific states to plot: Susceptible (1), Exposed (2), Symptomatic (4)
    selected_states = Dict(
        "Never" => 1,
        "Exposed" => 2,
        "Presymptomatic" => 3,
        "Symptomatic" => 4,
        "Asymptomatic" => 5,
        "Deaths" => 6
    )


 # Colors for each plot
    for (state, idx) in selected_states
        plt = plot(title="ExaEpi vs. Neural Reconstruction: $state Population Fit (Iter: $iter)",
                   xlabel="Time", ylabel="Population",
                   legend=:topleft, size=(800, 500),
                   titlefontsize=14, guidefontsize=12, tickfontsize=10)

        # Observed data: black/blue/red dots
        scatter!(plt, tdata, ode_data[idx, :],
                 label="ExaEpi Average",
                 markershape=:circle, color=:black, markersize=5)

        # Predicted data: matching dashed line
        plot!(plt, tdata, sol[idx, :], label="Reconstructions",
               linestyle=:solid, linewidth=2, color=:red)

        if lockdown_time !== nothing # Lockdown time: vertical dashed line
            vline!([lockdown_time], label="Lockdown Start", linestyle=:dash, color=:gray)
        end

        # Save to file
	savefig(plt, "$(OUTPUT_DIR)/u0_fitting_$(state)_iter_$(iter).png")

   display(plt)

    end
end

function plot_multiple_shoot(plt, preds, group_size)
    step = group_size - 1
    ranges = group_ranges(datasize, group_size)

    println("Total computed group ranges: ", length(ranges))
    #@assert length(ranges) <= 20 "More than 20 groups detected!  Check group segmentation logic."
    for (i, rg) in enumerate(ranges)
        #println("Plotting Group $i with range: ", rg)  # Debug: Print each group range

        plot!(plt, tdata[rg], preds[i][1, :]; legend=:topleft, markershape=:circle, label="Group $i")
    end
end


mkpath("$OUTPUT_DIR") 
loss_history = Float64[]  # Store loss values

# Early stopping parameters
patience = 1000  # Stop if no improvement in 1000 iterations
best_loss = Inf
no_improve_iters = 0

N = 1000  # Plot every 1000 iterations
callback = function (p,
l; doplot = true) 
    global iter, loss_history
    iter += 1

    display(l)  # Print loss to console

    # Store the current loss in our array
    push!(loss_history, l)
    if doplot && iter % 1000 == 0
        p_current = ComponentArray(p.u, pax) 

        # Re-calculate predictions inside the callback for plotting
        _, preds = multiple_shoot(p_current, ode_data, tdata, prob_node, loss_function, Tsit5(), group_size;
continuity_term)

        plt = scatter(tdata, ode_data[1, :]; label="Data")
        plot_multiple_shoot(plt, preds, group_size)

        # Save intermediate multiple shooting plots_multiple_shooting_main6
	savefig(plt, "$OUTPUT_DIR/multi_shooting_iter_$(iter).png")

        # Solve and plot u0 fitting with the current parameters
        sol = solve(prob_node, Tsit5(), p=p_current, saveat=tdata)
        denorm_ode_data = ode_data .* max_vals

  denorm_sol = sol .* max_vals

        plot_u0_fitting_periodic(denorm_sol, tdata, denorm_ode_data, u0, iter)

        save_predictions_csv(sol, tdata, ode_data; max_vals=max_vals, path="$OUTPUT_DIR/predictions.csv",
                         tag="iter_$(iter)")
    end

    return false  # Continue optimization
end

# ADAM Optimization First
using_lbfgs = false  # Flag to track optimizer
res_ms = Optimization.solve(optprob, ADAM(0.001); callback=callback, maxiters=1000)

# Store ADAM loss before switching to LBFGS
adam_loss = loss_history[end]
println(" ADAM Final Loss: ", adam_loss)

# Prepare for LBFGS
p_LBFGS_init = getdata(ComponentArray(res_ms.u, pax))  # Convert ADAM output to vector

# Reset loss history for LBFGS
loss_history = Float64[]

# Define new LBFGS optimization problem
optprob_LBFGS = Optimization.OptimizationProblem(optf, p_LBFGS_init)

# Run LBFGS
using_lbfgs = true  # Set flag for correct logging
lbfgs_iters = 50000
res_lbfgs = Optimization.solve(optprob_LBFGS, LBFGS(linesearch=BackTracking());
callback=callback, maxiters=lbfgs_iters)

# Store LBFGS loss
lbfgs_loss = loss_history[end]
println("LBFGS Final Loss: ", lbfgs_loss)

# Compare ADAM vs LBFGS Loss
println("Loss Before LBFGS: ", adam_loss, " | Loss After LBFGS: ", lbfgs_loss)

# Solve and Plot using LBFGS-optimized parameters
p_LBFGS_final = ComponentArray(res_lbfgs.u, pax)
sol_lbfgs = solve(prob_node, Tsit5(), p=p_LBFGS_final, saveat=tdata)

# Plot u₀ fitting after LBFGS
plot_u0_fitting_periodic(sol_lbfgs, tdata, ode_data, u0, iter)

# Plot Final Loss Curve
final_loss_plot = plot(loss_history, xlabel="Iteration", ylabel="Loss", title="Final Loss Curve (ADAM + LBFGS)", lw=2, label="Loss")
savefig(final_loss_plot, "$OUTPUT_DIR/final_loss_curve_LBFGS.png")
display(final_loss_plot)

# Final CSVs: normalized and denormalized (to match avg_output.dat scale)
save_predictions_csv(sol_lbfgs, tdata, ode_data;
                     path="$OUTPUT_DIR/predictions_final_normalized.csv")

save_predictions_csv(sol_lbfgs, tdata, ode_data; max_vals=max_vals,
		     path="$OUTPUT_DIR/predictions_final_denormalized.csv")


using Plots, Colors

function plot_final_comparison(sol, tdata, ode_data, u0, iter)
    plt = plot(title="Final Fit: Observed vs. Predicted (Iter $(iter))",
               xlabel="Time", ylabel="State Variables")

    colors = distinguishable_colors(length(u0))  # Unique colors for each state

    for i in 1:length(u0)
        color_i = colors[i]  # Assign unique color

       # Observed data (dots) - same color as predicted line
         scatter!(plt, tdata, ode_data[i, :],
                 label="Observed - State $i",
                 markershape=:circle,
                 markerstrokewidth=0.5,
                 markercolor=color_i,
                 markersize=5)

        plot!(plt, tdata, sol[i, :],
                label="Predicted - State $i",
                linestyle=:solid, linewidth=2,
                color=color_i)

    end

    # Save the final figure
    savefig(plt, "$OUTPUT_DIR/final_fit_iter_$(iter).png")
    display(plt)
end
plot_final_comparison(sol_lbfgs, tdata, ode_data, u0, iter)

function plot_final_on_multishoot(sol, tdata, ode_data, u0, iter; scatter_fraction=0.3)
    plt = plot(title="Final Fit Over Multiple Shooting Groups (Iter $(iter))",
               xlabel="Time", ylabel="State Variables")

    colors = distinguishable_colors(length(u0))  # Unique colors for each state

    # Plot subsampled multi-shooting groups (Scatter)
    for i in 1:length(u0)  # Loop over states
        color_i = colors[i]  # Assign unique color

        for (group_idx, rg) in enumerate(group_ranges(length(tdata), 5))
 # Adjust group_size
             sampled_indices = rg[1:round(Int, length(rg) * scatter_fraction)]

            scatter!(plt, tdata[sampled_indices], ode_data[i, sampled_indices],
                label=nothing,  # Removes redundant legend labels
                markerstrokewidth=0.3, alpha=0.6, color=color_i)
    end
    end

    # Overlay final predicted values (Solid Lines)
    for i in 1:length(u0)
        color_i = colors[i]
        plot!(plt, tdata, sol[i, :],
              label="Predicted - State $i",
              linewidth=2,
              color=color_i)
    end

    savefig(plt, "$OUTPUT_DIR/final_multishoot_overlay_iter_$(iter).png")
    display(plt)
end

plot_final_on_multishoot(sol_lbfgs, tdata, ode_data, u0, iter, scatter_fraction=0.3)

function plot_final_multiplot2(sol, tdata, ode_data, u0, iter;
scatter_fraction=0.3)
    plt = plot(title="Final Fit Over Multiple Shooting Groups (Iter $(iter))",
               xlabel="Time", ylabel="State Variables")

    colors = distinguishable_colors(length(u0))  # Unique colors for each state

    # Scatter observed data from multiple shooting groups
    for i in 1:length(u0)  # Loop over states
        color_i = colors[i]  # Assign unique color

        for (group_idx, rg) in
enumerate(group_ranges(length(tdata), 5))  # Adjust group_size
            sampled_indices = rg[1:min(round(Int, length(rg) * scatter_fraction), length(rg))]  # Avoid out-of-bounds
            scatter!(plt, tdata[sampled_indices], ode_data[i, sampled_indices],
                     label=nothing,  # Removes redundant legend labels
                     markerstrokewidth=0.3, alpha=0.6, color=color_i)

        end
    end


   # Overlay final predicted values (Solid Lines)
    for i in 1:length(u0)
         color_i = colors[i]
        plot!(plt, tdata, sol[i, :],
              label="Predicted - State $i",
              linewidth=2,
              color=color_i)
    end

    savefig(plt, "$OUTPUT_DIR/final_multishoot_overlay2_iter_$(iter).png")
    display(plt)
end

plot_final_multiplot2(sol_lbfgs, tdata, ode_data, u0, iter,
scatter_fraction=0.3)


anim = Plots.Animation()
iter = 0

callback = function (p, l, preds;
doplot = true)
    display(l)  # Print loss

    global iter
    iter += 1

    if doplot && iter % 10 == 0
        plt = scatter(tdata, ode_data[1, :]; label = "Data")
        plot_multiple_shoot(plt, preds, group_size)

        # Save to file instead of displaying
	savefig(plt, "$OUTPUT_DIR/multi_shooting_iter_$(iter).png")
    end

   return false
end
