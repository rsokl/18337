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

Base.:+(x::MyComplex,y::MyComplex) = MyComplex(x.real+y.real,x.imag+y.imag)
Base.:/(x::MyComplex,y::Int) = MyComplex(x.real/y,x.imag/y)
average(x::Vector{MyComplex}) = sum(x)/length(x)
@code_llvm average(arr)

sum(i for i in 1:100)


using Base.Threads

acc = 0
@threads for i in 1:10_000
    global acc
    acc += 1
end

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

