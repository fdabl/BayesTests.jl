module BayesTests

using Distributions, Gadfly,
      KernelDensity, Cubature, Mamba

export BinomTest, CorrelationTest, TTest,
       plot, update, posterior, ppv, hpd

include("common.jl")
include("TTest.jl")
include("BinomTest.jl")
include("CorrelationTest.jl")

end # module
