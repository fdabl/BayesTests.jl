include("BayesTest.jl")


type BinomTest <: BayesTest
    prior::Distribution
    data::Dict{AbstractString, Union{Int64, Float64}}
    BF::Float64 # BF_{01}
    posterior::Any # either distribution, or NaN
    params::Array{AbstractString, 2}

    function BinomTest(y::Integer, N::Integer; a = 1, b = 1)
        self = new()
        self.params = ["θ"]'
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
