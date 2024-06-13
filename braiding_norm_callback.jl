# Braiding following Beenakker's review 1907.06497
using QuantumDots
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using DiffEqCallbacks
includet("misc.jl")
## Get the majoranas
c = FermionBasis(1:2)
majorana_labels = 0:3
γ = MajoranaWrapper(c, majorana_labels)

## Give the value of the three deltas at time t in the three point majorana braiding protocol
function braiding_deltas(t, T, Δmax, Δmin, k, args...)
    Δ1 = Δtrajectory(t, T, Δmax, Δmin, k)
    Δ2 = Δtrajectory(t + T / 3, T, Δmax, Δmin, k)
    Δ3 = Δtrajectory(t + 2T / 3, T, Δmax, Δmin, k)
    return Δ1, Δ2, Δ3
end
function Δtrajectory(t, T, Δmax, Δmin, k, args...)
    dΔ = Δmax - Δmin
    tp = mod(t, T) - T / 12
    Δmin + dΔ * smooth_step(tp - 1T / 6, k) - dΔ * smooth_step(tp - 4T / 6, k)
end


## Couplings
const γ01 = Matrix(1.0im * γ[0] * γ[1])
const γ02 = Matrix(1.0im * γ[0] * γ[2])
const γ03 = Matrix(1.0im * γ[0] * γ[3])
const Ps = (γ01, γ02, γ03);
##
function H((T, Δmin, Δmax, k), t)
    Δs = braiding_deltas(t, T, Δmax, Δmin, k)
    sum(Δ * P for (Δ, P) in zip(Δs, Ps))
end
function norm_error(resid, u, p)
    resid[1] = norm(u) - 1
end
cb = ManifoldProjection(norm_error, resid_prototype=[0.0], autonomous=Val(true), isinplace=Val(true), save=false)
## Parameters for 
vacuumvec = zeros(ComplexF64, size(γ01, 1))
vacuumvec[1] = 1
u0 = vacuumvec # Initial state
u0 = u0 / norm(u0)
T = 1 # Total time for single braiding
k = 100 / T # Larger k means steeper steps for the couplings
Δmax = 1e2 / T
Δmin = Δmax * 1e-2
p = (T, Δmin, Δmax, k)
tspan = (0.0, 2T)
##
prob = ODEProblem(drho!, u0, tspan, p)
ts = range(0, tspan[2], 300)
deltas = stack([braiding_deltas(t, p...) for t in ts])'
plot(ts, deltas, label=["Δ1" "Δ2" "Δ3"], xlabel="t")

## Solve the system
@time sol = solve(prob, Tsit5(), saveat=ts, save_everystep=false, tstops=ts, reltol=1e-6, callback=cb);
plot(ts, 1 .- map(norm, sol), label="norm error", xlabel="t")

## lets measure the parities
measurements = Ps
plot(ts, [real(sol(t)'m * sol(t)) for m in measurements, t in ts]', legend=false, xlabel="t")