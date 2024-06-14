# Braiding following Beenakker's review 1907.06497
using QuantumDots
using LinearAlgebra
using Plots
using OrdinaryDiffEq
includet("misc.jl")
## Get the majoranas
c = FermionBasis(1:3, qn=QuantumDots.parity)
majorana_labels = 0:5
γ = MajoranaWrapper(c, majorana_labels)

## Couplings
P = parity_operators(γ)
Ps = (P[0, 1], P[0, 2], P[0, 3]);
## 
function H((T, Δmin, Δmax, k), t)
    Δs = braiding_deltas(t, T, Δmin, Δmax, k)
    sum(Δ * P for (Δ, P) in zip(Δs, Ps))
end

## Parameters
u0 = collect(first(eachcol(eigen(Hermitian(P[1, 4] + P[2, 5] - P[0, 3]), 1:1).vectors)))
T = 1 # Total time for single braiding
k = 50 / T # Larger k means steeper steps for the couplings
Δmax = 1e3 / T
Δmin = 1e-3 * 1 / T
p = (T, Δmin, Δmax, k)
tspan = (0.0, 2T)
##
prob = ODEProblem(drho!, u0, tspan, p)
ts = range(0, tspan[2], 300)
deltas = stack([braiding_deltas(t, p...) for t in ts])'
plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)

## Solve the system
@time sol = solve(prob, Tsit5(), saveat=ts, reltol=1e-12, tstops=ts);
plot(ts, [1 .- norm(sol(t)) for t in ts], label="norm error", xlabel="t")

##
parities = [(2, 3), (2, 4), (3, 5)]
measurements = map(p -> P[p...], parities)
plot(plot(ts, [real(sol(t)'m * sol(t)) for m in measurements, t in ts]', label=permutedims(parities), legend=true, xlabel="t"),
    plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=1), layout=(2, 1), lw=2, frame=:box)

##
prob_full = ODEProblem(drho!, Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)), tspan, p)
@time sol_full = solve(prob_full, Tsit5(), saveat=ts, reltol=1e-12, tstops=ts);
sol_full(2T)