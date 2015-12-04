using BayesTests
using Base.Test


binomBF = BinomTest(1, 1)
samples = posterior(binomBF; iter = 10000)
credible = interval(samples)

@test binomBF.BF == 1.0
@test_approx_eq_eps credible[1] .167 .1
@test_approx_eq_eps credible[2] .986 .1

update(binomBF, 3, 3)

corBF = CorrelationTest(1, 1; a = 1)
@test corBF.BF == 0.0

update(corBF, 0, 1)
@test corBF.BF == 0.0

x1 = collect(1:10)
x2 = collect(11:20)

ttestBF = TTest(x1, x2)
update(ttestBF, x1, x2)

ttestBF = TTest(x1, x2; typ = :twosample)
update(ttestBF, x1, x2)
