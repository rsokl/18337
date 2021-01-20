# Concurrency

@time sleep(2)

# Non-blocking concurrency:
# Runs instantly because spawns threads and never
# comes back to ask about completion
@time for i in 1:10
    @async sleep(2)
end



# Can make the loop-block itself a blocking operation;
# this should take 2 seconds
@time @sync for i in 1:10
    @async sleep(2)
end


# Can use @Threads.spawn:
#= Threads.@spawn expr

Create and run a Task on any available thread. To wait for the task to finish, call wait  
on the result of this macro, or call fetch to wait and then obtain its return value. =#


# Multithreading

using StaticArrays, BenchmarkTools

# If we want to thread the computation for each du-i then the following from
# poses a problem: each du-i is stack-allocated, and each thread's stack is local,
# so we cannot add threading to this form to build up [du1, du2, du3] from separate threads

function lorenz(u, p)
    α, σ, ρ, β = p
    @inbounds begin
        du1 = u[1] + α * (σ * (u[2] - u[1]))
        du2 = u[2] + α * (u[1] * (ρ - u[3]) - u[2])
        du3 = u[3] + α * (u[1] * u[2] - β * u[3])
    end
    @SVector [du1,du2,du3]
end

function solve_system_save!(u, f, u0, p, n)
    @inbounds u[1] = u0
    @inbounds for i in 1:length(u) - 1
        u[i + 1] = f(u[i], p)
    end
    u
end

p = (0.02, 10.0, 28.0, 8 / 3)

u = Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef, 1000)

@btime solve_system_save!(u, lorenz, @SVector([1.0,0.0,0.0]), p, 1000)


# In this form we forego making du a stack-allocated static vector, and instead
# allocate it to the heap, this way this form can be modified so that we can
# multithread the du-i computations and have each thread contribute its result
# to the shared heap
function lorenz!(du, u, p)
    α, σ, ρ, β = p
    @inbounds begin
        du[1] = u[1] + α * (σ * (u[2] - u[1]))
        du[2] = u[2] + α * (u[1] * (ρ - u[3]) - u[2])
        du[3] = u[3] + α * (u[1] * u[2] - β * u[3])
    end
end

function solve_system_save_iip!(u, f, u0, p, n)
    @inbounds u[1] = u0
    @inbounds for i in 1:length(u) - 1
        f(u[i + 1], u[i], p)
    end
    u
end

p = (0.02, 10.0, 28.0, 8 / 3)

# lives on the heap
u = [Vector{Float64}(undef, 3) for i in 1:1000]

@btime solve_system_save_iip!(u, lorenz!, [1.0,0.0,0.0], p, 1000)


using Base.Threads

# We can multithread it... but it is actually going to be much slower!
# the cost of spinning up new threads is 50ns per thread. Each one
# also represents an allocation, whereas before we required no allocations.
function lorenz_mt!(du, u, p)
    α, σ, ρ, β = p
    let du = du, u = u, p = p
        Threads.@threads for i in 1:3
            @inbounds begin
                if i == 1
                    du[1] = u[1] + α * (σ * (u[2] - u[1]))
                elseif i == 2
                    du[2] = u[2] + α * (u[1] * (ρ - u[3]) - u[2])
                else
                    du[3] = u[3] + α * (u[1] * u[2] - β * u[3])
                end
                nothing
            end
        end
    end
    nothing
end

p = (0.02, 10.0, 28.0, 8 / 3)
u = [Vector{Float64}(undef, 3) for i in 1:1000]

@btime solve_system_save_iip!(u, lorenz_mt!, [1.0,0.0,0.0], p, 1000)

3 * 50E-9 * 1000


# Multithreaded Parameter Searches

using Statistics

const _u_cache = Vector{typeof(@SVector([1., 0., 0.]))}(undef, 1000)

function compute_trajectory_mean(u0, p)
    solve_system_save!(_u_cache, lorenz, u0, p, 1000)
    return mean(_u_cache)
end

p = (0.02, 10.0, 28.0, 8 / 3)

@btime compute_trajectory_mean(@SVector([1., 0., 0.]), p)

# Conducting a parameter space
ps = [(0.02, 10., 28., 8 / 3) .* (1., rand(3)...) for i in 1:1000]

serial_out = map(p -> compute_trajectory_mean(@SVector([1., 0., 0.]), p), ps)
@btime serial_out = map(p -> compute_trajectory_mean(@SVector([1., 0., 0.]), p), ps)


# If we want to parallelize this then we need to anticipate the fact that
# each thread will need to manipulate its own preallocated _u_cache


const _u_cache_threads = [Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef, 1000) for i in 1:Threads.nthreads()]

function compute_trajectory_mean_mt(u0, p)
    _u_cache = _u_cache_threads[Threads.threadid()]
    solve_system_save!(_u_cache, lorenz, u0, p, 1000)
    return mean(_u_cache)
end

p
@btime compute_trajectory_mean_mt(@SVector([1., 0., 0.]), p)

function tmap(f, ps)
    out = Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef, length(ps))
    Threads.@threads for i in 1:length(ps)
        out[i] = f(ps[i])
    end
    return out
end

threaded_out = tmap(p -> compute_trajectory_mean_mt(@SVector([1.0,0.0,0.0]), p), ps)

serial_out - threaded_out

@btime serial_out = map(p -> compute_trajectory_mean(@SVector([1., 0., 0.]), p), ps)
@btime threaded_out = tmap(p -> compute_trajectory_mean_mt(@SVector([1.0,0.0,0.0]), p), ps)

# Task-Based Parallelism / Multithreading
isleep(i) = (sleep(i / 10); i)

# Will call sleep in order of executions
# this will be dominated by the longer tasks

# My computer can use 6 cores. And the loop gets chunked
# into four iterations per core. So the first core runs for:
sum(i/10 for i in 1:4)

# The last core runs for (9s)
sum(i/10 for i in 21:24)

# Thus the whole computation should take 9 seconds

function sleepmap_static()
    out = Vector{Int}(undef, 24)
    Threads.@threads for i in 1:24
        sleep(i / 10)
        out[i] = i
    end
    out
end
  

# Will dynamically run tasks to keep cores busy. Should
# take 24/10 - 2.4 s
function sleepmap_spawn()
    tasks = [Threads.@spawn(isleep(i)) for i in 1:24]
    out = [fetch(t) for t in tasks]
end
  
@btime sleepmap_static()
@btime sleepmap_spawn()
