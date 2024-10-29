# Braiding following Beenakker's review 1907.06497
using QuantumDots
using LinearAlgebra
using Plots
using OrdinaryDiffEq
includet("misc.jl")
## Get the majoranas
c = FermionBasis(1:2)
majorana_labels = 0:3
γ = MajoranaWrapper(c, majorana_labels)
## Couplings
P = parity_operators(γ)
Ps = (P[:M,:M̃], P[:M,:L], P[:M,:R]);
##
function H((T, Δmin, Δmax, k), t)
    Δs = braiding_deltas(t, T, Δmin, Δmax, k)
    sum(Δ * P for (Δ, P) in zip(Δs, Ps))
end

## Parameters
u0 = first(eachcol(eigen(Hermitian(P[:M̃,:L̃] + P[2, 5] - P[:M,:R]), 1:1).vectors))
T = 1000 # Total time for single braiding
k = 100 / T # Larger k means steeper steps for the couplings
Δmax = 1e3 / T
Δmin = 0Δmax / T
p = (T, Δmin, Δmax, k)
tspan = (0.0, 2T)
##
prob = ODEProblem(drho!, u0, tspan, p)
ts = range(0, tspan[2], 300)
deltas = stack([braiding_deltas(t, p...) for t in ts])'
plot(ts, deltas, label=["Δ1" "Δ2" "Δ3"], xlabel="t")

## Solve the system
@time sol = solve(prob, Tsit5(), saveat=ts, reltol=1e-12, tstops=ts);
sol(2T)
##
plot(ts, [1 .- norm(sol(t)) for t in ts], label="norm error", xlabel="t")

## lets measure the parities
measurements = Ps
plot(ts, [real(sol(t)'m * sol(t)) for m in measurements, t in ts]', legend=false, xlabel="t")