include("BayesTest.jl")
import Gadfly.plot, Mamba.hpd


type BinomTest <: BayesTest
    prior::Distribution
    data::Dict{AbstractString, Real}
    BF::Float64 # BF_{01}
    posterior::Any # either distribution, or NaN
    params::AbstractString

    function BinomTest(y::Integer, N::Integer; a = 1, b = 1)
        self = new()
        self.params = "θ"
        self.data = Dict("y" => y, "N" => N)
        self.prior = Distributions.Beta(a, b)
        self.posterior = Distributions.Beta(a + y, b + N - y)
        self.BF = pdf(self.prior, .5) / pdf(self.posterior, .5) # Savage-Dickey Trick
        return self
    end
end


function update(Model::BinomTest, y, N)
    BinomTest(Model.data["y"] + y, Model.data["N"] + N;
              a = Model.posterior.α, b = Model.posterior.β)
end


function posterior(Model::BinomTest; iter = 10000)
    return rand(Model.posterior, iter)
end


function hpd(samples::Array; low::Real = .025, up::Real = .975)
    ss = sort(samples); n = length(ss)
    return ss[Int(n * low)], ss[Int(n * up)]
end


function plot(Model::BinomTest)
    fun(x) = Distributions.pdf(Model.posterior, x)
    sup = Distributions.support(Model.posterior)
    Gadfly.plot(fun, sup.lb, sup.ub)
end


function plot(samples::Array)
    density = KernelDensity.kde(samples)
    f(x) = KernelDensity.pdf(density, x)
    Gadfly.plot(f, minimum(dist), maximum(dist))
end
