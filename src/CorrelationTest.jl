include("BayesTest.jl")


type CorrelationTest <: BayesTest
    prior::Distribution
    data::Dict{AbstractString, Integer}
    BF::Float64 # BF_{01}
    posterior::Any # either distribution, or NaN
    params::Array{AbstractString, 2}


    function CorrelationTest(n::Int64, r::Union{Int64, Float64}; a::Union{Int64, Float64} = 1)
        self = new()
        self.params = ["ρ"]'
        self.data = Dict("n" => n, "r" => r)
        self.prior = Distributions.Beta(a, a)
        self.BF = computeBF(n, r, a)
        self.posterior = NaN # not yet implemented
        return self
    end


    function computeBF(n::Int64, r::Union{Int64, Float64}, a::Union{Int64, Float64})

        if n < 2
            return 0
        end

        if n > 2 && abs(r) == 1
            return Inf
        end

        hypergeom(a, b, c, d) = a + b + c + d # dummy
        hyperterm = hypergeom((2 * n - 3) / 4, (2 * n - 1) / 4, (n + 2 * a) / 2, r^2)
        logterm = lgamma((n + 2 * a - 1) / 2) - lgamma((n + 2 * a) / 2) - lbeta(a, a)
        BF01 = √π * 2^(1 - a) * exp(logterm) * hyperterm

        return BF01
    end
end


function update(Model::CorrelationTest, r, n)
    CorrelationTest(r, n; a = Model.prior.α)
end
