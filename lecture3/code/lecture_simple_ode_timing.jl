using Flux
NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
           Dense(1,32,tanh),
           Dense(32,1),
           first) # Take first value, i.e. return a scalar
NNODE(1.0)

g(t) = t*NNODE(t) + 1f0

using Statistics
ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)

opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)

using BenchmarkTools
@btime Flux.train!(loss, Flux.params(NNODE), data, opt)  # 17.948 s (141805057 allocations: 7.76 GiB)