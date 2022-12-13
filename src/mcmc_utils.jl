using Distributions
using PyPlot
using Random
using Printf
using ConjugatePriors

# Redefinition of geometric distribution
include("GeometricNats.jl")

################################################################################
# Declare any composite types
################################################################################
mutable struct Results
    # sampled parameters
    θs::Vector{Any}

    # sampled truncations
    Ks::Vector{Int64}

    # associated sign σ(data, θ, K)
    σs::Vector{Float64}

    # number of negative signs ∑ 1(σ(x_i, θ, k) < 0)
    nns::Vector{Float64}

    # number of likelihood estimator calls
    nlike::Vector{Float64}

    # save the log target density at the last sampled values
    logPs::Vector{Float64}

    # name of results
    label::String
end

################################################################################
# Code for inference algorithm ("Algorithm 1")
################################################################################

function inference!(results, T, inputs, sample_K!, sample_θ!, log_density)
    """
    Generic inference function.

    Inputs:
    - results: a Results type containing initial values, sampled values, signs, and bookkeeping
    - T: number of MCMC iterations to run
    - inputs: a dictionary containing data, hyperparameters, etc.
    - sample_K!: a function that takes results, inputs, and log_density,
        updating results with a sample from the conditional target K | θ
    - sample_θ!: a function that takes results, inputs, and log_density,
        updating results with a sample from the conditional target θ | K
    - log_density: a function that takes (θ, data, γ) and returns a log_probability and a sign of that

    Results of inference are stored in results, and there is no return value.
    """

    for t in 1:T
        t % 100 == 0 && println("Iteration: $t")
        sample_K!(results, inputs, log_density)
        sample_θ!(results, inputs, log_density)
    end

end

################################################################################
# Code for conditional samplers
################################################################################

function sample_θ_RWMH!(results, inputs, log_density; sigma=1.0, Lp=nothing)
    # last value
    θ = results.θs[end]
    K = results.Ks[end]

    D = size(θ, 1); @assert D == 1; d=1

    θ_l = zeros(D)
    θ_r = zeros(D)
    θ_prime = zeros(D)

    nlike = 0; log_P = 0

    if Lp == nothing
        log_P, σ = log_density(θ[d], K, inputs)
        nlike += 1
    else
        log_P = Lp
    end

    # Propose new θ
    θ_prime[d] = rand(Normal(θ[d], sigma))

    # Compute log joint and get sign
    log_P_prime, σ_prime = log_density(θ_prime[d], K, inputs)

    # Compute MH ratio
    logR = logpdf(Normal(θ_prime[d], sigma), θ[d]) - logpdf(Normal(θ[d], sigma), θ_prime[d])
    logR += log_P_prime - log_P

    # Log acceptance probability
    logR = min(log(1), logR)

    # accept/reject proposal and save sample
    if log(rand()) < logR
        push!(results.θs, θ_prime)
        push!(results.σs, σ_prime)
    else
        push!(results.θs, θ)
        push!(results.σs, σ)
    end
end


function update_K_nothing!(results, inputs, log_density)
    # do nothing
    push!(results.Ks, results.Ks[end])
end


function update_K_RWMH!(results, inputs, log_density; Lp=nothing)
    """
    K: current value of truncation level K
    θ: current value of parameter θ
    log_density: joint
    """

    # get the most recent sampled values
    θ = results.θs[end]
    K = results.Ks[end]

    # geometric hyperparameter
    γ = inputs[:γ]

    # temporary
    d = 1

    if Lp == nothing
        log_P, _ = log_density(θ[d], K, inputs)
    else
        log_P = Lp
    end

    # propose a new candidate value
    if rand(Bernoulli(0.5))
        Knew = K + 1
    else
        Knew = K - 1
    end

    logR = logpdf(GeometricNats(γ), Knew) - logpdf(GeometricNats(γ), K)
    #logR = logpdf(Geometric(γ), Knew-1) - logpdf(Geometric(γ), K-1)
    logR += log_density(θ[d], Knew, inputs)[1] - log_density(θ[d], K, inputs)[1]

    R = min(1, exp(logR))

    # accept/reject proposal and save sample
    if rand() < R
        push!(results.Ks, Knew)
    else
        push!(results.Ks, K)
    end
end

################################################################################
# Code for RR and SS estimator
################################################################################
function like_estimator(θ, x, K, lf_like, inputs)
    """
    Compute RR estimator. This should work for general likelihoods. Assumes geometric(γ) distribution on K.

    Inputs:
    * θ: parameter
    * x: observation
    * K: truncation level.
    * lf_like: a function of (θ, x, k) that gives values in the sequence Lk
    * inputs: dict of input values
    """

    # geometric hyperparameter
    γ = inputs[:γ]

    w(K, k, γ) = 1 / (1 - cdf(Geometric(γ), k-1-1)) # the extra -1 is to get right cdf supp

    tot_sum = 0
    Lk1 = 0
    for k in 1:K
        # compute L_k
        Lk = lf_like(θ, x, k)

        # compute kth term of the sum
        tot_sum += exp(log(w(K, k, γ)) + log(Lk)) - exp(log(w(K,k,γ)) + log(Lk1))

        Lk1 = Lk
    end

    return tot_sum
end

function RR_estimator(θ, x, K, lf_like, inputs)
    """
    Compute RR estimator. Assumes geometric(γ) distribution on K.

    Inputs:
    * θ: parameter
    * x: observation
    * K: truncation level.
    * lf_like: a function of (θ, x, k) that gives values in the sequence Lk
    * inputs: dict of input values
    """

    # geometric hyperparameter
    γ = inputs[:γ]

    w(K, k, γ) = 1 / (1 - cdf(Geometric(γ), k-1-1)) # the extra -1 is to get right cdf supp

    tot_sum = 0
    Lk1 = 0

    for k in 1:K
        # compute L_k
        Lk = lf_like(θ, x, k; inputs)

        # compute kth term of the sum
        tot_sum += w(K, k, γ) * (Lk - Lk1)

        Lk1 = Lk
    end

    return tot_sum
end
