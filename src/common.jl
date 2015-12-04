include("BayesTest.jl")
import Base.mean, Base.median, Gadfly.plot


function posterior(Model::BayesTest; iter = 10000)
    samples = rand(Model.posterior, iter)
    return samples
end


function interval(samples; low = .025, up = .975)
    ss = sort(samples)
    n = length(ss)
    return ss[Int(n * low)], ss[Int(n * up)]
end


function plot(Model::BayesTest)
    if Model.posterior != false
        fun(x) = Distributions.pdf(Model.posterior, x)
        sup = Distributions.support(Model.posterior)
        Gadfly.plot(fun, sup.lb, sup.ub)
    end
end


function plot(samples::PosteriorSamples, pindex::Int64)
    dist = samples[:, pindex]
    density = KernelDensity.kde(dist)
    f(x) = KernelDensity.pdf(density, x)
    Gadfly.plot(f, minimum(dist), maximum(dist))
end
