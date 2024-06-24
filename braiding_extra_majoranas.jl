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
function H((T, Δmin, Δmax, k, ϵs, ζs, corr), t)
    Δs = braiding_deltas(t, T, Δmin, Δmax, k)

    Ham = sum(real(Δ) * P for (Δ, P) in zip(Δs, Ps))
    Ham += ϵs[1] * P[0, 1] + ϵs[2] * P[2, 4] + ϵs[3] * P[3, 5]
    Ham += imag(Δs[2]) * (ζs[1] * P[1, 2] + ζs[2] * P[0, 4]) + imag(Δs[3]) * (ζs[1] * P[1, 3] + ζs[3] * P[0, 5]) # First order in zeta
    Ham += -real(Δs[2]) * ζs[1] * ζs[2] * P[1, 4] - real(Δs[3]) * ζs[1] * ζs[3] * P[1, 5] # Second order in zeta
    # Introduce a correction term against perurbations from zeta 1, 3 and zeta 1, 2
    Δ23 = √( Δs[2]^2 + Δs[3]^2)
    Δ31 = √( Δs[3]^2 + Δs[1]^2)
    Δ12 = √( Δs[1]^2 + Δs[2]^2)
    Ham += -corr* ζs[1] * ζs[3] * Δ23*Δs[3]/Δ31 * P[2, 4]
    Ham += -corr* ζs[1] * ζs[2] * Δ23*Δs[2]/Δ12 * P[3, 5]
end
function mat_update!(A, u, (T, Δmin, Δmax, k, Ps), t)
    fill!(A, 0)
    for (Δ, P) in zip(braiding_deltas(t, T, Δmin, Δmax, k), Ps)
        A .+= -1im .* Δ .* P
    end
end
M = MatrixOperator(rand(ComplexF64, size(first(P)[2])...); (update_func!)=mat_update!)

## Parameters
u0 = collect(first(eachcol(eigen(Hermitian(P[0, 1] + P[2, 4] + P[3, 5]), 1:1).vectors)))
Δmax = 1
T = 1e3 / Δmax
k = 1e1
Δmin = 1e-6 * Δmax
ϵs = (0.0, 0.0, 0.0) # Energy overlaps between Majoranas ordered as ϵ01, ϵ24, ϵ35
ζ = 1e-2
ζs = (ζ, ζ, ζ) # Unwanted Majorana contributions within each island ordered as ζ01, ζ24, ζ35
correction = 0
p = (T, Δmin, Δmax, k, ϵs, ζs, correction)
tspan = (0.0, 2T)

##
prob = ODEProblem(drho!, u0, tspan, p)
ts = range(0, tspan[2], 1000)
deltas = stack([braiding_deltas(t, p...) for t in ts])'
plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)

## Solve the system
@time sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts);
#@time sol = solve(prob, LinearExponential(), saveat=ts, abstol=1e-6, reltol=1e-6, dt=1e-4);
plot(ts, [1 .- norm(sol(t)) for t in ts], label="norm error", xlabel="t")

##
parities = [(0, 1), (2, 4), (3, 5)] # (1, 4), (2, 5)
measurements = map(p -> P[p...], parities)
plot(plot(ts, [real(sol(t)'m * sol(t)) for m in measurements, t in ts]', label=permutedims(parities), legend=true, xlabel="t", ylims=(-1, 1) ),
    plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=1), layout=(2, 1), lw=2, frame=:box)

##
prob_full = ODEProblem(drho!, Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)), tspan, p)
@time sol_full = solve(prob_full, Tsit5(), saveat=ts, reltol=1e-12, tstops=ts);
sol_full(2T)
##
# Do a sweep over several zetas, solve the system for the final time t=2T and measure the parities
zetas = range(0, 1, length=100)
parities_arr = zeros(ComplexF64, length(zetas), length(measurements))
correction = 1

for (idx, ζ) in enumerate(zetas)
    p = (T, Δmin, Δmax, k, ϵs, (ζ, ζ, ζ), correction)
    prob = ODEProblem(drho!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=ts, reltol=1e-12, tstops=ts)
    parities_arr[idx, :] = [real(sol(2T)'m * sol(2T)) for m in measurements]
    println("ζ = $ζ, parities = $(parities_arr[idx, :])")
end
##
# Plot the parities as a function of the zetas
plot(zetas, real(parities_arr), label=permutedims(parities), xlabel="ζ", ylabel="Parity", lw=2, frame=:box)
## Do a sweep over the total braiding time T and the zetas and plot the parities
# Choose all energies and times in values of Deltamax
# Define T as the x axis and zeta as the y axis
Δmax = 1
Δmin = 1e-4 * Δmax
ϵs = Δmax * [0.0, 0.0, 0.0]
k = 1e1

gridpoints = 10
T_arr = range(1e2, 3e3, length=gridpoints) * 1/Δmax
zetas = range(0, 1, length=gridpoints)
parities_after_T_2D = zeros(ComplexF64, gridpoints, gridpoints, length(measurements))
parities_arr_2D = zeros(ComplexF64, gridpoints, gridpoints, length(measurements))
correction = 1

using Base.Threads

for (idx_T, T) in enumerate(T_arr)
    # Please write the above loop over zetas as parallelized loop below this Line
    Threads.@threads for idx_z in 1:gridpoints
        tspan = (0.0, 2T)
        ζ = zetas[idx_z]
        ts = range(0, tspan[2], 1000)
        p = (T, Δmin, Δmax, k, ϵs, (ζ, ζ, ζ), correction)
        prob = ODEProblem(drho!, u0, tspan, p)
        sol = solve(prob, Tsit5(), saveat=ts, reltol=1e-12, tstops=ts)
        parities_after_T_2D[idx_z, idx_T, :] = [real(sol(T)'m * sol(T)) for m in measurements]
        parities_arr_2D[idx_z, idx_T, :] = [real(sol(2T)'m * sol(2T)) for m in measurements]
        println("T = $T, ζ = $ζ, parities = $(parities_arr_2D[idx_T, idx_z, :])")
    end
end
##
# Plot the parities and parities_after_T (2, 4) as a continous colormap  

##
