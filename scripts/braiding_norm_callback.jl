# Braiding following Beenakker's review 1907.06497
using QuantumDots
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using DiffEqCallbacks
includet("misc.jl")
## Get the majoranas
c = FermionBasis(1:3; qn=QuantumDots.parity)
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
function mat_update!(A, u, (T, Δmin, Δmax, k), t)
    fill!(A, 0)
    for (Δ, P) in zip(braiding_deltas(t, T, Δmin, Δmax, k), Ps)
        A .+= -1im * Δ * P
    end
end
M = MatrixOperator(H((1, 1, 1, 1), 1); (update_func!)=mat_update!)
function norm_error(resid, u, p)
    resid[1] = norm(u) - 1
end
cb = ManifoldProjection(norm_error, resid_prototype=[0.0], autonomous=Val(true), isinplace=Val(true), save=false)
## Parameters
u0 = first(eachcol(eigen(Hermitian(P[1, 4] + P[2, 5] - P[0, 3]), 1:1).vectors))
T = 1 # Total time for single braiding
k = 100 / T # Larger k means steeper steps for the couplings
Δmax = 1e2 / T
Δmin = 1e-2 / T
p = (T, Δmin, Δmax, k)
tspan = (0.0, 2T)
##
prob = ODEProblem(M, u0, tspan, p)
# prob = ODEProblem(drho!, u0, tspan, p)
ts = range(0, tspan[2], 300)
deltas = stack([braiding_deltas(t, p...) for t in ts])'
plot(ts, deltas, label=["Δ1" "Δ2" "Δ3"], xlabel="t")

## Solve the system
@time sol = solve(prob, Tsit5(), saveat=ts, save_everystep=false, tstops=ts, reltol=1e-6);
@time sol_cb = solve(prob, Tsit5(), saveat=ts, save_everystep=false, tstops=ts, reltol=1e-6, callback=cb);
plot(ts, 1 .- map(norm, sol), label="norm error", xlabel="t");
plot!(ts, 1 .- map(norm, sol_cb), label="norm error with callback", xlabel="t")

## lets measure the parities
parities = [(1, 4), (2, 5), (0, 3)]
measurements = map(p -> P[p...], parities)
plot(plot(ts, [real(sol_cb(t)'m * sol_cb(t)) for m in measurements, t in ts]', label=permutedims(parities), legend=true, xlabel="t"),
    plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=1), layout=(2, 1), lw=2, frame=:box)

##
function norm_error_full(resid, u, p)
    resid .= map(norm, eachcol(u)) .- 1
end
cb_full = ManifoldProjection(norm_error_full, resid_prototype=0.0u0, autonomous=Val(true), isinplace=Val(true), save=false)
prob_full = ODEProblem(M, Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)), tspan, p, callback=cb_full)
@time sol_full = solve(prob_full, Tsit5(), saveat=ts, reltol=1e-12, tstops=ts);
blockdiagonal(sol_full(2T), c).blocks[2] #Even sector