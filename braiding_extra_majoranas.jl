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
P = Dict(map((kp) -> kp[1] => kp[2][2^2+1:end, 2^2+1:end], collect(P))); #Only take the even sector
Ps = (P[0, 1], P[0, 2], P[0, 3]);
## 
function H((T, Δmin, Δmax, k, Ps), t)
    Δs = braiding_deltas(t, T, Δmin, Δmax, k)
    sum(Δ * P for (Δ, P) in zip(Δs, Ps))
end
function mat_update!(A, u, (T, Δmin, Δmax, k, Ps), t)
    fill!(A, 0)
    for (Δ, P) in zip(braiding_deltas(t, T, Δmin, Δmax, k), Ps)
        A .+= -1im .* Δ .* P
    end
end
M = MatrixOperator(rand(ComplexF64, size(first(P)[2])...); (update_func!)=mat_update!)

## Parameters
u0 = collect(first(eachcol(eigen(Hermitian(P[2, 4] + P[3, 5] + P[0, 1]), 1:1).vectors)))
T = 1 # Total time for single braiding
k = 50 / T # Larger k means steeper steps for the couplings
Δmax = 1e3 / T
Δmin = 1e-3 * 1 / T
p = (T, Δmin, Δmax, k, Ps)
tspan = (0.0, 2T)
##
prob = ODEProblem(M, u0, tspan, p)
ts = range(0, tspan[2], 300)
deltas = stack([braiding_deltas(t, p...) for t in ts])'
plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)

## Solve the system
@time sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts);
#@time sol = solve(prob, LinearExponential(), saveat=ts, abstol=1e-6, reltol=1e-6, dt=1e-4);
plot(ts, [1 .- norm(sol(t)) for t in ts], label="norm error", xlabel="t")

##
parities = [(2, 3), (2, 4), (3, 5)]
measurements = map(p -> P[p...], parities)
plot(plot(ts, [real(sol(t)'m * sol(t)) for m in measurements, t in ts]', label=permutedims(parities), legend=true, xlabel="t"),
    plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=1), layout=(2, 1), lw=2, frame=:box)

##
prob_full = ODEProblem(drho!, Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)), tspan, p)
@time sol_full = solve(prob_full, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts);
sol_full(2T)