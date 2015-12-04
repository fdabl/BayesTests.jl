include("BayesTest.jl")


type CorrelationTest <: BayesTest
    prior::Distribution
    data::Dict{AbstractString, Real}
    BF::Float64 # BF_{01}
    posterior::Any # either distribution, or NaN
    params::Array{AbstractString, 2}


    function CorrelationTest(n::Integer, r::Real; a::Real = 1)
        self = new()
        self.params = ["ρ"]'
        self.data = Dict("n" => n, "r" => r)
        self.prior = Distributions.Beta(a, a)
        self.BF = computeBF(n, r, a)
        self.posterior = NaN # TODO: implement posterior
        return self
    end


    function computeBF(n::Integer, r::Real, a::Real)
        if n < 2 return NaN end
        if n > 2 && abs(r) == 1 return Inf end

        hypergeom(a, b, c, d) = a + b + c + d # dummy
        hyperterm = hypergeom((2 * n - 3) / 4, (2 * n - 1) / 4, (n + 2 * a) / 2, r^2)
        logterm = lgamma((n + 2 * a - 1) / 2) - lgamma((n + 2 * a) / 2) - lbeta(a, a)
        BF01 = √π * 2.0^(1 - 2*a) * exp(logterm) * hyperterm

        return BF01
    end
end
