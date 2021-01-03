# Demonstrating a naive higher order function

"""
`solve_system(f, u0, n, p)`

Solves the dynamical system:

``u_{n+1} = f(u_{n}, p)``

for N steps.
"""
function solve_system(f, u0, p, n)
    # Because any function in julia is its own individual type,
    # the compiler can actually specialize `solve_system` for the particular
    # `f` passed to it.
    # 
    # Furthermore, this means that the compiler has all the info it needs
    # about `f` in order to inline `f` here and perform type-inference.
    # Thus this naive implementation is actually well-optimized!
    u = u0
    #display(u)
    for i in 1:n - 1
        u = f(u, p)
        #display(u)
    end
    return u
end

f(u,p) = u^2 - p * u

# fixed points at: 
#     u = 0, u = p + 1
# derivatives at fixed points
#   df = 2u - p
#   -p,  2 + p

solve_system(f, -.9, 0.25, 1000)

using Plots
p = -1.9
t = collect(-1:0.001:p+1)
plot(t, t.^2 .- p .* t)
plot!(t, 2t .- p)

start_vals = -2:0.01:2
end_vals = []
for start in start_vals
    push!(end_vals, solve_system(f, start, p, 10))
end
#end_vals
end_vals[abs.(end_vals) .> 1000.] .= -2
plot(start_vals, end_vals)

solve_system(f, -1.001, p, 10)


f(0, p)
2*1.20 - p

using BenchmarkTools
@btime solve_system(f,1.251,0.25,10)

# Multidimensional system implementations
function lorenz(u, p)
    α,σ,ρ,β = p
    du1 = u[1] + α*(σ*(u[2]-u[1]))
    du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du3 = u[3] + α*(u[1]*u[2] - β*u[3])
    [du1,du2,du3]
  end


  p = (0.02,10.0,28.0,8/3)
  solve_system(lorenz,[1.0,0.0,0.0],p,1000)

  # Using a matrix to store all of these values is not necessarily very
  # efficient since we don't actually care about the memory being contiguous
  # and yet we have to spend time allocating it.
  # The cache is also hard to optimize
  function solve_system_save(f,u0,p,n)
    u = Matrix{eltype(u0)}(undef, 3, n)
    u[:, 1] = u0
    for i in 1:n-1
      u[:, i + 1] = f(@view(u[:, i]), p)
    end
    u
  end


to_plot = solve_system_save(lorenz, [1.0,0.0,0.0], p, 1000)

x = [to_plot[1, i] for i in 1:size(to_plot, 2)]
y = [to_plot[2, i] for i in 1:size(to_plot, 2)]
z = [to_plot[3, i] for i in 1:size(to_plot, 2)]

# Plot: the chaotic Lorenz attractor plotted in phase space
plot(x,y,z)

function solve_system_save_via_vector(f,u0,p,n)
    u = Vector{typeof(u0)}(undef, 1)
    u[1] = u0
    for i in 1:n-1
      push!(u, f(u[i], p))
    end
    u
end

to_plot = solve_system_save_via_vector(lorenz, [1.0,0.0,0.0], p, 1000)

x = [to_plot[i][1] for i in 1:length(to_plot)]
y = [to_plot[i][2] for i in 1:length(to_plot)]
z = [to_plot[i][3] for i in 1:length(to_plot)]
z

@btime solve_system_save_via_vector(lorenz, [1.0,0.0,0.0], p, 1000)
@btime solve_system_save(lorenz, [1.0,0.0,0.0], p, 1000)

function lorenz!(du,u,p)
    α,σ,ρ,β = p
    du[1] = u[1] + α*(σ*(u[2]-u[1]))
    du[2] = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du[3] = u[3] + α*(u[1]*u[2] - β*u[3])
end

function solve_system_no_alloc(f!, u0, p, n)
    # create work buffer
    u = copy(u0)
    du = similar(u)
    # non-allocating loop
    for _ in 1:n-1
        f!(du, u, p)
        u, du = du, u
    end
    return u
end
u0 = [1.0,0.0,0.0]
@btime solve_system_no_alloc(lorenz!, u0, p, 1000)
u0


# Return to saving but use static StaticArrays
using StaticArrays
using BenchmarkTools

function lorenz(u, p)
    α,σ,ρ,β = p
    @inbounds begin
        du1 = u[1] + α*(σ*(u[2]-u[1]))
        du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
        du3 = u[3] + α*(u[1]*u[2] - β*u[3])
    end
    @SVector [du1,du2,du3]
  end

function solve_system_save(f,u0,p,n)
    u = Vector{typeof(u0)}(undef,n)
    @inbounds u[1] = u0
    @inbounds for i in 1:n-1
        u[i+1] = f(u[i],p)
    end
    return u
end

# this is so fast!
u0 = @SVector [1.0, 0.0, 0.0]
@btime solve_system_save(lorenz, u0, p, 1000)
