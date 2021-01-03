using Flux

NN = Chain(Dense(10, 32, tanh),
           Dense(32, 32, tanh),
           Dense(32, 5)
           )

loss() = sum(abs2, sum(abs2, NN(rand(10)) .- 1) for i in 1:100)

p = params(NN)

# data generation is "built-in" to the loss, thus we feed an empty iterator
Flux.train!(loss, p, Iterators.repeated((), 10000), ADAM(0.1))

loss()

NN(rand(10))