# Code examples from lecture notes

using BenchmarkTools
using StaticArrays
using LinearAlgebra
using Plots

# Demonstrating row-major vs column-major memory access.
A = rand(100, 100)
B = rand(100, 100)
C = rand(100, 100)

function row_major!(C, A, B)
    for i in 1:100, j in 1:100
        C[i, j] = A[i, j] + B[i, j]
    end
end


function col_major!(C, A, B)
    for j in 1:100, i in 1:100
        C[i, j] = A[i, j] + B[i, j]
    end
end

@btime row_major!(C, A, B)
@btime col_major!(C, A, B)

###############################
# Heap Allocations and Speed #
##############################

function sum_with_alloc!(C, A, B)
    for j in 1:100, i in 1:100
        # cannot prove at compile time that `val` will 
        # always have a given size, thus a heap-allocation
        # must take place
        val = [A[i, j] + B[i, j]]
        C[i, j] = val[1]
    end
end

function sum_no_alloc!(C,A,B)
    for j in 1:100, i in 1:100
      val = A[i,j] + B[i,j]
      C[i,j] = val
    end
  end
  
@btime sum_with_alloc!(C, A, B)
@btime sum_no_alloc!(C, A, B)  # 30x faster



function sum_with_static_alloc!(C, A, B)
    for j in 1:100, i in 1:100
        # cannot prove at compile time that `val` will 
        # always have a given size, thus a heap-allocation
        # must take place
        val = @SVector [A[i, j] + B[i, j]]
        C[i, j] = val[1]
    end
end

@btime sum_with_static_alloc!(C, A, B)

function unfused_sum(A, B, C)
    tmp = A .+ B
    tmp .+= C
    return tmp
end

fused_sum(A, B, C) = A .+ B .+ C

@btime unfused_sum(A, B, C)
@btime fused_sum(A, B, C)

D = similar(A)
inplaced_fused_sum(D, A, B, C) = D .= A .+ B .+ C

@btime inplaced_fused_sum(D, A, B, C)

# slicing copies by default
@btime x = A[:, :]
@btime @view A[:, :]


#############################
# Heap Allocation Asymptotics#
##############################

function alloc_timer(n)
    A = rand(n, n)
    B = rand(n, n)
    C = similar(A)
    t1 = @belapsed $A .* $B
    t2 = @belapsed $C .= $A .* $B
    return t1, t2
end

unzip(a) = (getfield.(a, x) for x in fieldnames(eltype(a)))

sizes = 2 .^ (2:11)
alloc, no_alloc = unzip(alloc_timer.(sizes))
sizes
plot(sizes, collect(alloc), label="=", yscale=:log10, xscale=:log10, legend=:bottomright, title="Micro-Optimizations Matter for BLAS1")
plot!(sizes, collect(no_alloc), label=".=")

function alloc_matmul_timer(n)
    A = rand(n, n)
    B = rand(n, n)
    C = similar(A)
    t1 = @belapsed $A * $B
    t2 = @belapsed mul!($C, $A, $B)
    return t1, t2
end

sizes = 2 .^ (2:7)
alloc, no_alloc = (collect(i) for i in unzip(alloc_matmul_timer.(sizes)))

plot(sizes, alloc, label="=", yscale=:log10, xscale=:log10, legend=:bottomright, title="Micro-Optimizations Don't Matter for Large MatMul")
plot!(sizes, no_alloc, label=".=")