using MajoranaBraiding
using QuantumDots
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using ProgressMeter
using StaticArrays
using Base.Threads

## Get the majoranas
c = FermionBasis(1:3, qn=QuantumDots.parity)
N = length(keys(c))
majorana_labels = 0:5
γ = MajoranaWrapper(c, majorana_labels)
use_static_arrays = true
inplace = !use_static_arrays
mtype, vtype = if use_static_arrays && inplace
    MMatrix{2^(N - 1),2^(N - 1)}, MVector{2^(N - 1)}
elseif use_static_arrays && !inplace
    SMatrix{2^(N - 1),2^(N - 1)}, SVector{2^(N - 1)}
else
    Matrix, Vector
end
## Couplings
P = parity_operators(γ, p -> (mtype(p[2^(N-1)+1:end, 2^(N-1)+1:end])));
## Parameters
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[:M,:M̃] + P[:L, :L̃] + P[:R, :R̃]), 1:1).vectors))))
Δmax = 1
T = 1e3 / Δmax
k = 1e1
Δmin = 1e-6 * Δmax
ϵs = (0.0, 0.0, 0.0) # Energy overlaps between Majoranas ordered as ϵ01, ϵ24, ϵ35
ζ = 0.1
ζs = (ζ, ζ, ζ) # Unwanted Majorana contributions within each island ordered as ζ01, ζ24, ζ35
tspan = (0.0, 2T)
ramp = RampProtocol([2, 1 / 3, 1] .* Δmin, [1 / 3, 1 / 2, 1] .* Δmax, T, k)
p = (ramp, ϵs, ζs, 1, 1, P)
H = ham_with_corrections
# H! = ham_with_corrections!
M = get_op(H, H!, p)
ts = range(0, tspan[2], 1000)

##

function cost_function(x, t)
    vals = eigvals(H((ramp, ϵs, ζs, x, 0, P), t))
    return vals[2] - vals[1]
end

costs1 = [cost_function(x, T / 3) for x in range(0, 2, length=1000)]
plot(log10.(costs1), xlabel="correction strength", ylabel="energy gap")

##
using Optim

results = Float64[]
alg = BFGS()
for t in ts
    f(x) = cost_function(only(x), t)
    initial = length(results) > 0 ? results[end] : 1.0
    result = optimize(f, [initial], alg, Optim.Options(time_limit=10 / length(ts)))
    push!(results, only(result.minimizer))
end
plot(ts, results, ylims=(0.5, 1.1))
##
using Interpolations

int = cubic_spline_interpolation(ts, results)
plot(ts, int(ts))
plot!(ts, results)

function optimized_corrmax(H, (ramp, ϵs, ζs, P), ts; alg=BFGS())
    results = Float64[]
    function cost_function(x, t)
        vals = eigvals(H((ramp, ϵs, ζs, x, 0, P), t))
        return vals[2] - vals[1]
    end
    for t in ts
        f(x) = cost_function(only(x), t)
        initial = length(results) > 0 ? results[end] : 1.0
        result = optimize(f, [initial], alg, Optim.Options(time_limit=1 / length(ts)))
        push!(results, only(result.minimizer))
    end
    return linear_interpolation(ts, results)
end