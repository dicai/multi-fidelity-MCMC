using Distributions
import Distributions: pdf, cdf, logpdf
import Base.rand
using PyPlot
import Distributions: @check_args
import Distributions: @distr_support

"""
Redefine the (parts of the) Geometric distribution (that we need) to be on {1,2,...}
instead of {0,1,...}.
"""
struct GeometricNats{T<:Real} <: DiscreteUnivariateDistribution
    p::T

    function GeometricNats{T}(p::T) where {T <: Real}
        new{T}(p)
    end
end

function GeometricNats(p::Real; check_args::Bool=true)
    @check_args GeometricNats (zero(p) < p < one(p))
    return GeometricNats{typeof(p)}(p)
end

GeometricNats() = GeometricNats{Float64}(0.5)

@distr_support GeometricNats 1 Inf

function pdf(d::GeometricNats, x::Real)
    return pdf(Geometric(d.p), x - 1)
end

function logpdf(d::GeometricNats, x::Int)
    return logpdf(Geometric(d.p), x - 1)
end

function cdf(d::GeometricNats, x::Int)
    return cdf(Geometric(d.p), x - 1)
end

function rand(d::GeometricNats, N::Int)
    return rand(Geometric(d.p), N) .+ 1
end

function rand(d::GeometricNats)
    return rand(Geometric(d.p)) + 1
end
