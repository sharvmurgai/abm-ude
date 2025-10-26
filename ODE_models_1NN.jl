function SEInsIsIaDR!(du, u, p, t, p_true;
                        scale_NN::Bool=false)
   
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

    S, E, Ins, Is, Ia, D, R = u

    pNN = p.pNN

    if !(t isa Float64)
        t = ReverseDiff.value(t)
    end

    output_NN = NN_model.model([E/N0, Ins/N0, Is/N0, Ia/N0, D/N0, R/N0], pNN, _state_var)[1]

    Kappa   = abs(output_NN[1])

    if scale_NN
        Kappa = Kappa * scale_Kappa
    end

    num = eta_a * Ins + Is + eta_a * Ia
    dem = S + E + Ins + Is + Ia + R
    lambda = p_trans * Kappa * num / dem

    T_ins = T_In - T_E
    T_s = T_E + T_i - T_In

    du[1] = dS = -lambda * S
    du[2] = dE = lambda * S - E / T_E
    du[3] = dIns = (1.0 - fa) * E / T_E - Ins / T_ins
    du[4] = dIs = Ins / T_ins - Is / T_s
    du[5] = dIa = fa * E / T_E - Ia / T_i
    du[6] = dD = (1.0 - fr) * Is / T_s
    du[7] = dR = fr * Is / T_s + Ia / T_i
end
