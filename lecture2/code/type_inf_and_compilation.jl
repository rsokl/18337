using InteractiveUtils

# Demonstrating type inference through functions
f(x, y) = x + y
@code_llvm f(2, 5)
@code_llvm f(2.0, 5.0)

function g(x,y)
    a = 4
    b = 2
    c = f(x,a)
    d = f(b,c)
    f(d,y)
  end


@code_llvm g(2, 5)
@code_warntype g(2, 5)

# Mixtures of types leads to promotions
@code_llvm f(2, 5.0)


##################
# Type Stability #
##################

# Breaking type stability
function h(x, y) # output is Union{Int,Float64} 
    out = x + y
    rand() < 0.5 ? out : Float64(out)  # Python version: `out if rand() < 0.5 else Float64(out)``
end

@code_warntype h(2, 5)



#####################
# Multiple Dispatch #
#####################
ff(x::Int, y::Int) = 2x + y
ff(x::Float64, y::Float64) = x / y

@show ff(2, 5)
@show ff(2.0, 5.0)

@which +(2.0, 5)


# fallback method still specializes on the inputs!
ff(x::Number, y::Number) = x + y

@code_llvm ff(2.0, 5)




##########
# isbits #
##########

struct MyComplex
    real::Float64
    imag::Float64
  end

isbitstype(MyComplex)

struct ParameterizedComplex{T}
    real::T
    imag::T
end

# the parameterized type itself does not guarantee that an
# instance will be isbits
isbitstype(ParameterizedComplex)
# but once an instance is created against isbits values, everything
# is a-ok
isbits(ParameterizedComplex(1.0, 2.0))

struct MySlowComplex
    real
    imag
  end

struct MySlowComplex{AbstractFloat}
    real::AbstractFloat
    imag::AbstractFloat
  end