using Mamba
include("BayesTest.jl")


type TTest <: BayesTest

    typ::Symbol
    prior::Distribution
    data::Dict{AbstractString, Any}
    BF::Float64 # BF_{01}
    posterior::Any # either distribution, or NaN
    params::Array{AbstractString, 2}


    function TTest(x1, x2; typ = :paired, scale::Real = √2/2)
        self = new()
        self.typ = typ;
        self.params = ["μ", "σ", "δ"]'
        self.data = Dict("x1" => x1, "x2" => x2)
        self.prior = Distributions.Cauchy(0, scale)
        typ == :paired ? PairedSamplesTTest(self) : TwoSampleTTest(self)
    end


    function PairedSamplesTTest(self::TTest)
        x1 = self.data["x1"]
        x2 = self.data["x2"]
        @assert length(x1) == length(x2) "Number of observations in each group must be equal"
        n = length(x1)
        df = n - 1

        t = mean(x1 - x2) / (√var(x1 - x2) / √n)
        self.BF = computeBF(t, n, df, self.prior.σ)
        return self
    end


    function TwoSampleTTest(self::TTest)
        x1 = self.data["x1"]
        x2 = self.data["x2"]
        s1 = var(x1); s2 = var(x2)
        n1 = length(x1); n2 = length(x2)
        n = Int((n1 + n2) / 2) # not sure, because BF doesn't do equal sample size

        pooled = √(s1/n1 + s2/n2)
        t = (mean(x1) - mean(x2)) / pooled
        df = (s1/n1 + s2/n2)^2 / ( (s1/n1)^2 / (n1 - 1) + (s1/n2)^2 / (n2 - 1) )
        self.BF = computeBF(t, n, df, self.prior.σ)
        return self
    end


    function computeBF(t::Real, n::Real, df::Real, γ::Real)
        f(g) = ((1 + n*g)^(-1/2) * (1 + t^2 / df*(1 + n*g))^(n-2) *
                (2π)^(-1/2) * g^(-3/2) * exp(-γ^2 / (2*g)))

        num = γ * hquadrature(f, 0.00001, 10)[1]
        denom = (1 + t^2/df)^(-n/2)
        return num / denom
    end
end


function update(Model::TTest, x1, x2)
    TTest(vcat(Model.data["x1"], x1),
          vcat(Model.data["x2"], x2),
          scale = Model.prior.σ)
end


function posterior(Model::TTest)
    model = Mamba.Model(
        x1 = Stochastic(1, (mu1, s1) ->  Normal(mu1, s1), false),
        x2 = Stochastic(1, (mu2, s2) ->  Normal(mu2, s2), false),
        mu1 = Stochastic(() -> Normal(0, 100)),
        mu2 = Stochastic(() -> Normal(0, 100)),
        s1 = Stochastic(() -> Uniform(0, 100)),
        s2 = Stochastic(() -> Uniform(0, 100)),
        δ = Logical((mu1, mu2, s1, s2) -> (mu2 - mu1) / ((s1 + s2) / 2))
    )

    scheme = [Mamba.NUTS([:mu1, :mu2, :s1, :s2])]
    line = Dict{Symbol, Any}(:x1 => Model.data["x1"], :x2 => Model.data["x2"])
    inits = [Dict{Symbol,Any}(
               :x1 => line[:x1],
               :x2 => line[:x2],
               :s1 => rand(Uniform(0, 100)),
               :s2 => rand(Uniform(0, 100)),
               :mu1 => rand(Normal(0, 100)),
               :mu2 => rand(Normal(0, 100))
             ) for i in 1:3]

    Mamba.setsamplers!(model, scheme)
    sim = Mamba.mcmc(model, line, inits, 5000, burnin = 500, chains = 2)
end
