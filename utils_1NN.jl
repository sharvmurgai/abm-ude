using ComponentArrays, Lux, Random, Distributions

nonneg(x::Real) = x >= zero(x)
rbf(x) = exp.(-(x .^ 2))
sigmoid(x) = 1 ./ (1 .+ exp.(-x))

#=============================#
#= Process data =#
    #=============================#

function process_data2(df, model)
    # if model == "SEInsIsIaDR"
    #     selected_df = df[:, ["Day", "Exposed", "Presymptomatic", "Symptomatic", "Asymptomatic", "Deaths", "Immune"]]
    # elseif model == "SEInsQIsIaDR"
    #     selected_df = df[:, ["Day", "Exposed", "Presymptomatic", "Quarantined", "Symptomatic", "Asymptomatic", "Deaths"]]
    # end
    #selected_df = df[:, ["Day", "Exposed", "Presymptomatic", "Symptomatic", "Asymptomatic", "Immune", "Deaths"]]
    #selected_df = df[:, ["Day", "Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic", "Immune", "Deaths"]]
    #selected_df = df[:, ["Day", "Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic", "Immune", "Deaths"]]
   
    selected_df = df[:, ["Day", "Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic", "Deaths", "Immune"]]
#     Day Never Infected Immune Deaths Hospitalized Ventilated ICU Exposed Asymptomatic Presymptomatic Symptomatic CumulativeDeaths MaxInfected
#     0 6.81511092e6 5.27 0.0 0.0 0.0 0.0 0.0 5.18 0.04 0.05 0.0 0.0 5.27
#     1 6.81511088e6 5.31 0.0 0.0 0.04 0.01 0.0 4.99 0.1 0.22 0.0 0.0 5.31
# ##

    println("Selected Columns Order: ", names(selected_df))
    println("First Row of `df`: ", df[1, ["Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic", "Deaths", "Immune"]])
    println("First Row of `selected_df` (without Day): ", selected_df[1, 2:end])


##
    tdata = convert(Vector{Float64}, selected_df.Day)
    data = Matrix{Float64}(selected_df[:, 2:end])  # Select only 6 states
    data = data'

    data_n_days = length(selected_df.Day)
    train_n_days = data_n_days
    tspan = (tdata[1], tdata[end])

    u0 = Array{Float64}(selected_df[1, 2:end])  # Ensure it matches the network input
    println("u0: ", u0)
    println("First Row of `selected_df` (without Day): ", selected_df[1, 2:end])

    println("DEBUG: u0 values = ", u0)
    println("DEBUG: typeof(u0) = ", typeof(u0))
    println("DEBUG: eltype(u0) = ", eltype(u0))

    # Runtime debug
    #DEBUG: u0 values = [5.18, 0.05, 0.0, 0.04, 0.0, 0.0]
    #DEBUG: typeof(u0) = Vector{Float64}
    #DEBUG: eltype(u0) = Float64

    return selected_df, tdata, tspan, train_n_days, data, u0
end

function process_data_march2(df, model)
    if model == "SEInsIsIaDR"
        correct_order = ["Day", "Immune", "Deaths", "Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic"]
    elseif model == "SEInsQIsIaDR"
        correct_order = ["Day", "Immune", "Deaths", "Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic"]
    end

    # Reorder `df` to match the original dataset order
    selected_df = df[:, correct_order]

    # Extract time series data
    tdata = convert(Vector{Float64}, selected_df.Day)
    data = Matrix{Float64}(selected_df[:, 2:end])'  # Transpose for correct shape

    # Initial conditions `u0`
    u0 = Array{Float64}(selected_df[1, 2:end])  

    # Debug Output
    println("Corrected Column Order: ", correct_order)
    println("First row of selected_df: ", selected_df[1, :])
    println("u0 values: ", u0)

    # Print first 5 rows of the original dataset
    println("First 5 rows of original dataset:")
    display(first(df, 5))

    # Print first 5 rows of selected_df (after reordering columns)
    println("First 5 rows of selected_df:")
    display(first(selected_df, 5))

    println("Original dataset - first 5 rows (selected columns):")
    display(df[:, ["Day", "Immune", "Deaths", "Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic"]][1:5, :])

    println("selected_df - first 5 rows:")
    display(selected_df[1:5, :])

    return selected_df, tdata, (tdata[1], tdata[end]), length(selected_df.Day), data, u0
end

function process_data(df, model)
    # if model == "SEInsIsIaDR"
    #     correct_order = ["Day", "Immune", "Deaths", "Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic"]
    # elseif model == "SEInsQIsIaDR"
    #     correct_order = ["Day", "Immune", "Deaths", "Exposed", "Asymptomatic", "Presymptomatic", "Symptomatic"]
    # end
    if model == "SEInsIsIaDR"
        correct_order =  ["Day", "Never", "Exposed", "Presymptomatic", "Symptomatic", "Asymptomatic", "Deaths"]
    elseif model == "SEInsQIsIaDR"
        correct_order = ["Day", "Never", "Exposed", "Presymptomatic", "Quarantined", "Symptomatic", "Asymptomatic", "Deaths"]
    end

    selected_df = df[:, correct_order]
    tdata = convert(Vector{Float64}, selected_df.Day)
    data = Matrix{Float64}(selected_df[:, 2:end])'
    u0 = Array{Float64}(selected_df[1, 2:end])
   
    max_vals = zeros(size(data, 1))  # Create a vector to store the max for each state

    for i in 1:size(data, 1)
        max_val = maximum(data[i, :])
        if max_val != 0
            data[i, :] .= data[i, :] ./ max_val
            u0[i] = u0[i] / max_val
        end
        max_vals[i] = max_val  # Store the max (even if 0)
    end


    # #Normalization Code: Normalize each state variable by its maximum
    # for i in 1:size(data, 1)
    #     max_val = maximum(data[i, :])
    #     if max_val != 0  # Avoid division by zero
    #         data[i, :] .= data[i, :] ./ max_val
    #         u0[i] = u0[i] / max_val
    #     end
    # end
   
    # Debug output for verification
    println("Normalized u0 values: ", u0)
   
    data_n_days = length(selected_df.Day)
    tspan = (tdata[1], tdata[end])
    train_n_days = data_n_days

    return selected_df, tdata, tspan, train_n_days, data, u0, max_vals
end

#= Multiple Shooting Loss Function =#
function multiple_shooting_loss(θ, train_data, tsteps, prob_node, group_size; continuity_term=200.0)
    step = group_size - 1
    ranges = group_ranges(length(tsteps), group_size)

    loss = 0.0
    prev_end_state = nothing

    for rg in ranges
        tsub = tsteps[rg]
        _prob = remake(prob_node, tspan=(tsub[1], tsub[end]), p=θ)
        _sol = solve(_prob, Tsit5(), saveat=tsub)

        if size(_sol, 2) == length(tsub)
            loss += sum(abs2, train_data[:, rg] .- _sol)
        else
            loss += 1.0 * 1e6  # Penalize failed solves
        end

        # Continuity loss: enforce smooth transition between segments
        if prev_end_state !== nothing
            loss += continuity_term * sum(abs2, prev_end_state .- _sol[:, 1])
        end

        prev_end_state = _sol[:, end]
    end

    return loss
end

#= Setting up parameters values =#
function set_params(rng_seed, model;
                    est_all::Bool=false,
                    return_NN::Bool=true,
                    scale_NN::Bool=false,
                    return_rng::Bool=false,
                    num_inputs::Vector{Int64}=[7])
    rng = StableRNG(rng_seed)
    if est_all == true
        if model == "SEInsIsIaDR"
            fixed_pars = Float64[
                rand(Gamma(5.46, 2.72), 1)[1], # T_E: Latent period
                rand(Gamma(6.9, 3.31), 1)[1], # T_In: Incubation period
                rand(Normal(15.3, 1.9), 1)[1], # T_i: Infectious period
                rand(Uniform(0.3, 1.0), 1)[1], # eta_a: Scaling factor for Asymptomatic Infectiousness
                rand(Normal(3.7, 0.005), 1)[1]/100.0, # Probability of transmission
            ]
        elseif model == "SEInsQIsIaDR"
            fixed_pars = Float64[
                rand(Gamma(5.46, 2.72), 1)[1], # T_E: Latent period
                rand(Gamma(6.9, 3.31), 1)[1], # T_In: Incubation period
                rand(Normal(15.3, 1.9), 1)[1], # T_i: Infectious period
                rand(Uniform(0.3, 1.0), 1)[1], # eta_a: Scaling factor for Asymptomatic Infectiousness
                rand(Normal(3.7, 0.005), 1)[1]/100.0, # Probability of transmission
                0.5, # Percent self isolation
            ]
        end
    else
        if model == "SEInsIsIaDR"
            fixed_pars = Float64[5.0, 7.0, 10.0, 0.5, 3.7/100.0]
        elseif model == "SEInsQIsIaDR"
            fixed_pars = Float64[5.0, 7.0, 10.0, 0.5, 3.7/100.0, 0.5]
        end
    end

    est_pars = Float64[rand(Normal(35.3, 5.5), 1)[1]/100.0, rand(Normal(90.0, 4.5), 1)[1]/100.0]

    # NN_model = Lux.Chain(
    #     Lux.Dense(num_inputs[1], 20, relu),
    #     Lux.Dense(20, 20, relu),
    #     Lux.Dense(20, 20, rbf),
    #     Lux.Dense(20, 1, rbf)
    # )
    # NN_model = Lux.Chain(
    # Lux.Dense(num_inputs[1], 32, tanh),  # Wider first layer
    # Lux.Dense(32, 64, swish),  # Swish improves smooth activation
    # Lux.BatchNorm(64),  # Normalization helps with stability
    # Lux.Dense(64, 64, tanh),  
    # Lux.Dense(64, 32, swish),  
    # Lux.ResidualBlock(Lux.Dense(32, 32, tanh)),  # Skip connection for deeper learning
    # Lux.Dense(32, 16, swish),  
    # Lux.Dense(16, 1, identity)  # Output without activation
    # )
    NN_model = Lux.Chain(
    Lux.Dense(num_inputs[1], 32, swish),   # Wider first layer for better feature extraction
    Lux.Dense(32, 64, swish),  
    Lux.Dense(64, 32, swish),  # Reduce complexity but keep capacity
    Lux.ResidualBlock(Lux.Dense(32, 32, tanh)),  # Skip connection for smoother updates
    Lux.Dense(32, 16, swish),  
    Lux.Dense(16, 1, identity)  # Output without activation
    )


    pNN, state_var = Lux.setup(rng, NN_model)
    NN = ((model=NN_model, pars=pNN, layer_states=state_var))

    scaler_pars = scale_NN ? Float64[rand(Normal(55, 4.5), 1)[1]] : nothing
    return return_NN ? (fixed_pars, est_pars, NN, scaler_pars, rng) : (fixed_pars, est_pars, nothing, scaler_pars, rng)
end


#= Multiple Shooting UDE Loss =#
function UDE_loss_fn(θ, train_data, tsteps, prob_node, group_size, continuity_term)
    return multiple_shooting_loss(θ, train_data, tsteps, prob_node, group_size; continuity_term=continuity_term)
end;


#= Train Function for Multi-Shooting =#
function train(p_ALL, train_data, tsteps, prob_node, cb;
                optimizer=PolyOpt(),
                maxiters::Int64=1000,
                ode_lb=nothing,
                ode_ub=nothing,
                group_size::Int64=3,
                continuity_term::Float64=200.0,
                kwargs...)

    adtype = Optimization.AutoZygote()
   
    optf = Optimization.OptimizationFunction(
        (x, p_ALL) -> UDE_loss_fn(x, train_data, tsteps, prob_node, group_size, continuity_term),
        adtype
    )

    p_ALL = ComponentVector{Float64}(p_ALL)
    if !isnothing(ode_lb) && !isnothing(ode_ub)
        _lb = [ode_lb; repeat([-Inf], length(p_ALL)-length(ode_lb))]
        _ub = [ode_ub; repeat([Inf],  length(p_ALL)-length(ode_ub))]
        optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p_ALL), lb=_lb, ub=_ub)
    else
        optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p_ALL))
    end

    res = Optimization.solve(optprob, optimizer; callback=cb, maxiters=maxiters, kwargs...)
    return res
end
