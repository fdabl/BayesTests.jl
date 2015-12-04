module BayesTests

using Distributions, Gadfly,
      KernelDensity, Cubature

export BinomTest, CorrelationTest, TTest,
       plot, update, interval, posterior

include("common.jl")
include("TTest.jl")
include("BinomTest.jl")
include("CorrelationTest.jl")

end # module
