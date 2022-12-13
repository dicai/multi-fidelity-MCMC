function log_density_estimator(θ, K, inputs)
    """
    This is a model-specific function. Define the sequence of low fidelity
    likelihoods inside this function.

    Inputs: θ, K, inputs, results

    Returns the log of the full likeilhood and the sign.
    """

    # geometric hyperparameter
    γ = inputs[:γ]
    data = inputs[:data]

    L_k(θ, x, k) = pdf(Normal(θ, 1 + 2/k^2), x)

    # construct RR estimator
    hat_L_k(θ, K, inputs, x) = like_estimator(θ, x, K, L_k, inputs)

    # log prob of prior
    lp = logpdf(Normal(0, 1), θ)

    like_signs = []

    # compute likelihood over all data, sign correction
    for i in 1:length(data)
        like = hat_L_k(θ, K, inputs, data[i])
        push!(like_signs, sign(like))
        if sign(like) == -1
            lp += log(abs(like) + 1e-20)
        else
            lp += log(like + 1e-20)
        end
    end

    inputs[:curr_like_signs] = like_signs

    return lp, prod(like_signs)
end

function log_density_estimator_single(θ, K, inputs)
    """
    This is a model-specific function. Define the sequence of low fidelity
    likelihoods inside this function.

    Inputs: θ, K, inputs, results

    Returns the log of the full likeilhood and the sign.
    """

    # geometric hyperparameter
    γ = inputs[:γ]
    data = inputs[:data]

    L_k(θ, x, k) = pdf(Normal(θ, 1 + 2/k^2), x)

    # construct RR estimator for the L_k defined above
    hat_L_k(θ, K, inputs, x) = SS_estimator(θ, x, K, L_k, inputs)

    # log prob of prior
    lp = logpdf(Normal(0, 1), θ)

    like_signs = []

    # compute likelihood over all data, sign correction
    for i in 1:length(data)
        like = hat_L_k(θ, K, inputs, data[i])
        push!(like_signs, sign(like))
        if sign(like) == -1
            lp += log(abs(like) + 1e-20)
        else
            lp += log(like + 1e-20)
        end
    end

    inputs[:curr_like_signs] = like_signs

    return lp, prod(like_signs)
end

function log_density_true(θ, K, inputs)
    data = inputs[:data]
    log_pr = logpdf(Normal(0, 1), θ)
    like_signs = []
    for i in 1:length(data)
        like = pdf(Normal(θ, σ0), data[i])
        push!(like_signs, sign(like))
        log_pr += log(abs(like))
    end
    return log_pr, prod(like_signs)
end

function log_density_biased(θ, K, inputs)
    L_k(θ, x, k) = pdf(Normal(θ, 1 + 2/k), x)

    data = inputs[:data]
    log_pr = logpdf(Normal(0, 1), θ)
    like_signs = []
    for i in 1:length(data)
        like = L_k(θ, data[i], K)
        push!(like_signs, sign(like))
        log_pr += log(abs(like))
    end
    return log_pr, prod(like_signs)
end

const σ0 = 1.0
function generate_data(N)
    t = rand(Normal(0,1))
    return t, rand(Normal(t, σ0), N)
end

