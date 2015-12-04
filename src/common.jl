using Mamba
include("BayesTest.jl")
import Base.mean, Base.median


function ppv(samples::Mamba.ModelChains)
    predict(samples) # TODO: implement posterior predictive p values
end
