using Flux

NN = Chain(Dense(10, 32, x -> max(0, x)), 
            Dense(32, 32, x -> max(0, x)), 
            Dense(32, 5, x -> max(0, x))
            )

NN(rand(10, 40))


##########################
# Defining a Dense Layer #
##########################

struct MyDense{F,S <: AbstractArray,T <: AbstractArray}
    W::S
    b::T
    σ::F
end

# Defining outer constructor
function MyDense(in::Integer, out::Integer, σ=identity,
    init_W=rand, init_b=zeros)
    return Dense(init_W(out, in), init_b(out), σ)
end

# Defining functor - equivalent to __call__
# a.k.a defining a callable struct
function (a::MyDense)(x::AbstractArray)
    return a.σ.(a.W * x .+ b)
end

d = MyDense(2, 3)

d(rand(2, 5))

##################
# Defining Chain #
##################

struct MyChain{T <: Tuple}
    layers::T
    # Defines an inner constructor, this permits *args style init.
    # Note that `xs...` in this context is "slurp",
    # whereas in a call it is "splat"
    # `new` is a special function available to inner constructors
    MyChain(xs...) = new{typeof(xs)}(xs)
end

# Uses recursion 
# - first definition is base case; note that `Tuple{}` is the type
# specific for an empty tuple!
# Note that the recursion enables a "zero-cost abstraction" 
applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(fs[2:end], first(fs)(x))

(c::MyChain)(x) = applychain(c.layers, x) 

MyNN = MyChain(MyDense(10, 32, x -> max(0, x)), 
            MyDense(32, 32, x -> max(0, x)), 
            MyDense(32, 5, x -> max(0, x))
            )

MyNN(rand(10, 40))

