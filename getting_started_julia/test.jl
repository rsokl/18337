# Permits the script to dictate that the local environment should be used.
# This helps to ensure reproducibility
cd(string(@__DIR__) * "\\getting_started_julia")
using Pkg
Pkg.activate(".")


using ForwardDiff, FiniteDiff

f(x) = 2x^2 + x
ForwardDiff.derivative(f, 2.0)
FiniteDiff.finite_difference_derivative(f, 2.0)
A = 2+2

using PkgTemplates
t = PkgTemplates.Template(user="RyanSoklaski")
t("Demo1")
