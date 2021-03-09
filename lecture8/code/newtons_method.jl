using StaticArrays, ForwardDiff

function newton_step(f, xo)
    J = ForwardDiff.jacobian(f, xo)
    δ = J \ f(xo)  # solves linear system
    return xo - δ
end

function newton(f, xo)
    x = xo
    for i in 1:10
        x = newton_step(f, x)
        @show x
    end
    return x
    
end


ff(xx) = ( (x, y) = xx;  SVector(x^2 + y^2 - 1, x - y) )

xo = SVector(3.0, 5.0)
x = newton(ff, xo)
