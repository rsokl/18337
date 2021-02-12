# How to do SIMD

struct MyComplex
    real::Float64
    imag::Float64
end

# This struct satisfies the "isbits" conditions of being immutable, and 
# consisting only of primitive/isbits types. Thus it can have a definite layout
# in memory
isbitstype(MyComplex)

# for this reason `arrs` is represented in memory as
# [real1, imag1, real2, imag2,...]
arr = [MyComplex(rand(), rand()) for i in 1:100]


# Let's contrast this with a struct of arrs
struct MyComplexes
    real::Vector{Float64}
    imag::Vector{Float64}
end

arrs2 = MyComplexes([rand() for i in 1:100], [rand() for i in 1:100])

using InteractiveUtils

# This reveals that llvm is creating small vectors and is calling
# vector-parallel instructions on them 

Base.:+(x::MyComplex,y::MyComplex) = MyComplex(x.real + y.real, x.imag + y.imag)
Base.:/(x::MyComplex,y::Int) = MyComplex(x.real / y, x.imag / y)
average(x::Vector{MyComplex}) = sum(x) / length(x)
@code_llvm average(arr)

sum(i for i in 1:100)


using Base.Threads

# loop-level parallelism
# - assumes iterations are independent
acc = 0
@threads for i in 1:10_000
    global acc
    acc += 1
end

# By contrast, `@spawn` could be used to perform task-based parallelism,
# which is more expressive & adaptive but usually requires modification 
# of information flow.

# tally wont match expected value because read/writes will be
# out of sync
acc


acc = Atomic{Int64}(0)
@threads for i in 1:10_000
    atomic_add!(acc, 1)
end

# we can ensure that only one thread can access the heap-allocated
# value at a time. However this, of course, has big performance implications
acc

using BenchmarkTools

# Julia exposes various lower-level heap control mechanisms
# via locks


# SpinLock is non-reentrant and thus will deadlock if it encounters
# consecutive lock states without an interceding unlock.
# Fast, but can be risky
const acc_lock = Ref{Int64}(0)
const splock = SpinLock()

function f1()
    @threads for i in 1:10_000
        lock(splock)
        acc_lock[] += 1
        unlock(splock)
    end
end

f1()
acc_lock
@btime f1()  # ~  146.700 μs

# ReentrantLock will not deadlock if it encounters multiple locks,
# although each lock must still get unlocked. Is slower though
const rsplock = ReentrantLock()
const acc_lock = Ref{Int64}(0)
function f2()
    @threads for i in 1:10_000
        lock(rsplock)
        acc_lock[] += 1
        unlock(rsplock)
    end
end

@btime f2()  # ~580.600 μs


# Atomics are faster than the above locks
acc2 = Atomic{Int64}(0)

function g()
    @threads for i in 1:10_000
        atomic_add!(acc2, 1)
    end
end

@btime g()  # ~91.300 μs

# But fastest of all is serial...?!?!
# Julia is too smart and can see that it just needs to
# add 10,000. This is why the serial loop seems to optimal
const acc_s = Ref{Int64}(0)

function h()
    global acc_s
    for i in 1:10_000
        acc_s[] += 1
    end
end

@btime h()  # ~3.475 μs


# We can force non-optimized performance by making the
# "length" of the sum non-constant
non_const_len = 10000

function h3()
    global acc_s
    global non_const_len
  # Note that if we made a type-declaration here,
  #       len2::Int64 = non_const_len
  # then the global could be used in a type-stable manner
  # and the loop could be optimized out once again.
    len2 = non_const_len
    for i in 1:len2
        acc_s[] += 1
    end
end

@btime h3()


# GPU Computing

# Distributed computing

## Explicit Memory Handling
using Distributed

# adds 4 worker processes to master control
addprocs(4)
@everywhere f(x) = x.^2 # Define this function on all processes

#  non-blocking but returns a future
t1 = remotecall(f, 1, randn(10))
t2 = remotecall(f, 2, randn(10))

# blocking call to retrieve value
xsq1 = fetch(t1)
xsq2 = fetch(t2)

## Task-Based Parallelism
using Dagger

add1(value) = value + 1
add2(value) = value + 2
combine(a...) = sum(a)

p = delayed(add1)(4)
q = delayed(add2)(p)
r = delayed(add1)(3)
s = delayed(combine)(p, q, r)

@assert collect(s) == 16
