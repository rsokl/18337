# x'' = -k x + 0.1 sin(x)
using Flux
using Plots
using DifferentialEquations
k = 1
force(dx, x, k, t) = -k * x + 0.1sin(x)
prob = SecondOrderODEProblem(force, 1.0, 0.0, (0.0, 10.0), k)
sol = solve(prob)

# it looks like `sol` can be indexed into to retrieve the solved
# velocity and position data
plot(sol, label=["Velocity" "Position"])

# Suppose we want to predict the force on the spring at each displacement: F(x)
# Let's say we have 4 measurements of x, x', and x''
plot_t = 0:1e-2:10
data_plot = sol(plot_t)
position_plot = [state[2] for state in data_plot]
force_plot = [force(state[1], state[2], k, 0) for state in data_plot]

# Generate the dataset
# `sol` can also be called like a function
#
# Note: the sampling used here is absolutely critical to the
# behavior seen in this exercise! These time values happen
# to sample the oscillator at nearly identical positions.
# This aliasing means that little useful information can be
# learned from the data.
#
# If you change the sampling so that this aliasing does not
# occur, then you actually learn this force (which is basically
# just a line on this domain...) from these few measurements really well!
#
# So why, then does training against data drawn from -kx work here?
# Well, the domain we are on means that `0.1 sin(x)` is practically
# a constant, so really we are learning to fit -kx + b.
# The first loss function basically just asks our NN to fit NN(0) = b...
# and then the "physics informed" loss term asks the NN to fit N(x) = -kx
# so uhh.. it basically just works out nicely because of this sampling.
#
# If you _don't_ have aliasing here, then
t = 0:3.7:10 # 0:2.5:10

dataset = sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1], state[2], k, t) for state in sol(t)]
force_data

plot(plot_t, force_plot, xlabel="t", label="True Force")
scatter!(t,force_data,label="Force Measurements")


# let's try to train a NN to learn the force function from these measurements
NNForce = Chain(x -> [x],
           Dense(1, 32, tanh),
           Dense(32, 1),
           first)

loss() = sum(abs2, NNForce(position_data[i]) - force_data[i] for i in 1:length(position_data))
loss()

# training naively
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () # callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(loss())
    end
end
display(loss())
Flux.train!(loss, Flux.params(NNForce), data, opt; cb=cb)

# naively training this NN leads to a function that matches only those 
# measurements (... not totally true; changing the sampling scheme of
# the measurements fixes this near-aliasing and the resulting solution
# is already very good)
learned_force_plot = NNForce.(position_plot)

plot(plot_t, force_plot, xlabel="t", label="True Force")
plot!(plot_t, learned_force_plot, label="Predicted Force")
scatter!(t, force_data, label="Force Measurements")


pos_domain = -0.9:0.001:0.9
true_f = [force(0, p, k, 0) for p in pos_domain]
learned_f = NNForce.(pos_domain)

# This is a much more honest depiction of what was learned...
plot(pos_domain, true_f, xlabel="x", label="True Force")
plot!(pos_domain, learned_f, label="Predicted Force")


# Simplified model of the spring might be F(x) = -kx
force2(dx,x,k,t) = -k * x
prob_simplified = SecondOrderODEProblem(force2, 1.0, 0.0, (0.0, 10.0), k)
sol_simplified = solve(prob_simplified)
plot(sol,label=["Velocity" "Position"])
plot!(sol_simplified,label=["Velocity Simplified" "Position Simplified"])

plot_t = 0:1e-2:10
data_plot = sol_simplified(plot_t)
position_plot_simplified = [state[2] for state in data_plot]

NNForce = Chain(x -> [x],
           Dense(1, 32, tanh),
           Dense(32, 1),
           first)

# Let's regularize the fitting by telling the NN that it should solve F(x) = -kx
random_positions = [2rand() - 1 for i in 1:100] # random values in [-1,1]
loss_ode() = sum(abs2, NNForce(x) - (-k * x) for x in random_positions)
loss_ode()

# Minimizing `loss_ode` solves that particular ode, so we can provide this as
# a weighted term in our objective
λ = 0.1
loss() = sum(abs2, NNForce(position_data[i]) - force_data[i] for i in 1:length(position_data))
composed_loss() = loss() + λ * loss_ode()

# training this physics-regularized ODE
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () # callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(composed_loss())
    end
end
display(composed_loss())
Flux.train!(composed_loss, Flux.params(NNForce), data, opt; cb=cb)

learned_force_plot = NNForce.(position_plot)
# learned_force_plot = NNForce.(position_plot_simplified)

# This is kind of a weird plot... it is piping in x(t)
# which is derived from the true form of F(x), which we 
# do not actually have access to with just the approx of -kx
# and those 4 data points
# When you change the positions of the samples to eliminate aliasing, 
# it does maybe a bit worse!
plot(plot_t, force_plot, xlabel="t", label="True Force")
plot!(plot_t, learned_force_plot, label="Predicted Force")
scatter!(t, force_data, label="Force Measurements")


pos_domain = -0.9:0.001:0.9
true_f = [force(0, p, k, 0) for p in pos_domain]
learned_f = NNForce.(pos_domain)

# This is a much more honest depiction of what was learned...
plot(pos_domain, true_f, xlabel="x", label="True Force")
plot!(pos_domain, learned_f, label="Predicted Force")

#plot!(pos_domain, 0.1 .* sin.(pos_domain))