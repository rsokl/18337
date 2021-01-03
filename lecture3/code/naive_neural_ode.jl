# Solving: u' = cos(2 pi t)  w/ u0 = 1.
# True solution: sin(2 pi t) / (2 pi) + 1
using Statistics
using Flux

NNODE = Chain(x -> reshape(x, 1, length(x)), Dense(1, 32, tanh), Dense(32, 1), x -> reshape(x, length(x)))
NNODE([1. 2.])

# encoding the boundary condition
g(t) = 1f0 .+ t .* NNODE(t)

# Loss: Sum[(dg/dt - cos(2 pi t)) ^ 2]
# Note that we actually need to backprop through the derivative of our
# function, so we will (naively) express it using finite-difference.
ϵ = sqrt(eps(Float32))

#loss() = mean(abs2(((g(t + ϵ) - g(t - ϵ)) / (2ϵ)) - cos(2π * t)) for t in 0:1f-2:1f0)
loss(t) = mean(abs2.(((g(t .+ ϵ) .- g(t .- ϵ)) ./ (2ϵ)) .- cos.(2π .* t)))

# Train our solution
opt = Flux.Descent(0.01)
data = collect(0:1f-2:1f0)
iter = 0
function cb2(t) # callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(loss(t))
    end
end
display(loss(data))

using BenchmarkTools
# implementation from lecture notes is 17.948 s (141805057 allocations: 7.76 GiB)
@btime Flux.train!(loss, Flux.params(NNODE), Iterators.repeated(data, 5000), opt)  # 1.397 s (4780058 allocations: 1.13 GiB)
@btime loss(data)
using Plots

# Plot the learned solution
t = 0:0.001:1.0
plot(t, g(collect(t)), label="NN")
plot!(t, 1 .+ sin.(2π .* t) ./ (2π), label="True Solution")
