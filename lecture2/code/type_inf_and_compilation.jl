using InteractiveUtils
using BenchmarkTools
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

struct MyComplex2{AbstractFloat}
    real::AbstractFloat
    imag::AbstractFloat
end

isbitstype(MyComplex)
isbitstype(MyComplex2)

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
isbits(MyComplex2(2.0, 1.0))

struct MySlowComplex
    real
    imag
  end
Base.:+(a::MyComplex,b::MyComplex) = MyComplex(a.real+b.real,a.imag+b.imag)
Base.:+(a::MyComplex,b::Int) = MyComplex(a.real+b,a.imag)
Base.:+(b::Int,a::MyComplex) = MyComplex(a.real+b,a.imag)

Base.:+(a::MyComplex2,b::MyComplex2) = MyComplex2(a.real+b.real,a.imag+b.imag)
Base.:+(a::MyComplex2,b::Int) = MyComplex2(a.real+b,a.imag)
Base.:+(b::Int,a::MyComplex2) = MyComplex2(a.real+b,a.imag)

Base.:+(a::ParameterizedComplex,b::ParameterizedComplex) = ParameterizedComplex(a.real+b.real,a.imag+b.imag)
Base.:+(a::ParameterizedComplex,b::Int) = ParameterizedComplex(a.real+b,a.imag)
Base.:+(b::Int,a::ParameterizedComplex) = ParameterizedComplex(a.real+b,a.imag)

struct MySlowComplex2  # If the struct were parameterized, it would be capable of isbits!
    real::AbstractFloat
    imag::AbstractFloat
  end

isbits(MySlowComplex(1.0, 2.0))
isbits(MySlowComplex2(1.0, 2.0))

Base.:+(a::MySlowComplex,b::MySlowComplex) = MySlowComplex(a.real+b.real,a.imag+b.imag)
Base.:+(a::MySlowComplex,b::Int) = MySlowComplex(a.real+b,a.imag)
Base.:+(b::Int,a::MySlowComplex) = MySlowComplex(a.real+b,a.imag)

Base.:+(a::MySlowComplex2,b::MySlowComplex2) = MySlowComplex2(a.real+b.real,a.imag+b.imag)
Base.:+(a::MySlowComplex2,b::Int) = MySlowComplex2(a.real+b,a.imag)
Base.:+(b::Int,a::MySlowComplex2) = MySlowComplex2(a.real+b,a.imag)

@code_warntype g(MySlowComplex(1.0,1.0), MySlowComplex(1.0,1.0))
@code_warntype g(MySlowComplex2(1.0,1.0), MySlowComplex2(1.0,1.0))

a = MyComplex(1.0,1.0)
b = MyComplex(2.0,1.0)
@btime g(a,b)

a = MyComplex2(1.0,1.0)
b = MyComplex2(2.0,1.0)
@btime g(a,b)

a = ParameterizedComplex(1.0,1.0)
b = ParameterizedComplex(2.0,1.0)
@btime g(a,b)

a = MySlowComplex(1.0,1.0)
b = MySlowComplex(2.0,1.0)
@btime g(a,b)

a = MySlowComplex2(1.0,1.0)
b = MySlowComplex2(2.0,1.0)
@btime g(a,b)