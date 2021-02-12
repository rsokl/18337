using DifferentialEquations

# Can speed up by turning off bounds checking
# and by allocating `du` as a static array
function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
end

# initial condition
u0 = [1.0,0.0,0.0]

# timespan of interest
tspan = (0.0, 100.0)   

# parameters for out particular system
p = (10.0, 28.0, 8 / 3)

prob = ODEProblem(lorenz!, u0, tspan, p)

sol = solve(prob)

using Plots
plot(sol)
