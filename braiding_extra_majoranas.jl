# Braiding following Beenakker's review 1907.06497
using QuantumDots
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using ProgressMeter
using StaticArrays
includet("misc.jl")
## Get the majoranas
c = FermionBasis(1:3, qn=QuantumDots.parity)
majorana_labels = 0:5
γ = MajoranaWrapper(c, majorana_labels)
use_static_arrays = true
## Couplings
N = length(keys(c))
P = parity_operators(γ)
P = use_static_arrays ? Dict(map((kp) -> kp[1] => SMatrix{2^(N - 1),2^(N - 1)}(kp[2][2^(N-1)+1:end, 2^(N-1)+1:end]), collect(P))) : Dict(map((kp) -> kp[1] => kp[2][2^(N-1)+1:end, 2^(N-1)+1:end], collect(P))); #Only take the even sector

## 
function H((T, Δmin, Δmax, k, ϵs, ζs, corr, P), t, α=1)
    Δs = braiding_deltas(t, T, Δmin, Δmax, k)
    Δs = braiding_deltas_new(t, T, Δmin, Δmax, k/T, true)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ31 = √(Δs[3]^2 + Δs[1]^2)
    Δ12 = √(Δs[1]^2 + Δs[2]^2)
    Ham = similar(first(P)[2])
    Ham = α * (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
               ϵs[1] * P[0, 1] + (ϵs[2] - corr * ζs[1] * ζs[3] * Δ23 * abs(Δs[3])/ Δ31) * P[2, 4] + (ϵs[3] - corr * ζs[1] * ζs[2] * Δ23 * Δs[2] / Δ12) * P[3, 5] +
               imag(Δs[2]) * (ζs[1] * P[1, 2] + ζs[2] * P[0, 4]) + imag(Δs[3]) * (ζs[1] * P[1, 3] + ζs[3] * P[0, 5]) +
               -real(Δs[2]) * ζs[1] * ζs[2] * P[1, 4] - real(Δs[3]) * ζs[1] * ζs[3] * P[1, 5] )
               #- corr* ζs[1] * ζs[3] * Δs[2] * P[0, 1] + corr * ζs[1] * ζs[3] * Δs[3] * Δs[1] * P[0, 2]
    return Ham
end
function H!(Ham, (T, Δmin, Δmax, k, ϵs, ζs, corr, P), t, α=1)
    Δs = braiding_deltas(t, T, Δmin, Δmax, k)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ31 = √(Δs[3]^2 + Δs[1]^2)
    Δ12 = √(Δs[1]^2 + Δs[2]^2)
    @. Ham = α * (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
                  ϵs[1] * P[0, 1] + (ϵs[2] - corr * ζs[1] * ζs[3] * Δ23 * Δs[3] / Δ31) * P[2, 4] + (ϵs[3] - corr * ζs[1] * ζs[2] * Δ23 * Δs[2] / Δ12) * P[3, 5] +
                  imag(Δs[2]) * (ζs[1] * P[1, 2] + ζs[2] * P[0, 4]) + imag(Δs[3]) * (ζs[1] * P[1, 3] + ζs[3] * P[0, 5]) +
                  +real( - Δs[2]) * ζs[1] * ζs[2] * P[1, 4] + real(Δs[3]) * ζs[1] * ζs[3] * P[1, 5])
    return Ham
end


## Parameters
u0 = collect(first(eachcol(eigen(Hermitian(P[0, 1] + P[2, 4] + P[3, 5]), 1:1).vectors)))
println("u0 = $u0")
use_static_arrays && (u0 = MVector{2^(N - 1)}(u0))
Δmax = 1
T = 3e3 / Δmax
k = 2e2
Δmin = 1e-6 * Δmax
ϵs = (0.0, 0.0, 0.0) # Energy overlaps between Majoranas ordered as ϵ01, ϵ24, ϵ35
ζ = 0.57
ζs = (ζ, 0*ζ, 1*ζ) # Unwanted Majorana contributions within each island ordered as ζ01, ζ24, ζ35
correction = 1
p = (T, Δmin, Δmax, k, ϵs, ζs, correction, P)
tspan = (0.0, 2T)
function mat_update(A, u, p, t)
    H(p, t, 1im)
end
function mat_update!(iHam, u, p, t)
    H!(iHam, p, t, 1im)
    return iHam
end
M = use_static_arrays ? MatrixOperator(H(p, 0, 1im); update_func=mat_update) : MatrixOperator(H(p, 0, 1im); (update_func!)=mat_update!)
##
prob = ODEProblem{!use_static_arrays}(M, u0, tspan, p)
ts = range(0, tspan[2], 1000)
deltas = stack([braiding_deltas(t, p...) for t in ts])'
deltas = stack([braiding_deltas_new(t, T, Δmin, Δmax, k/T, true) for t in ts])'
plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)

## Solve the system
@time sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts);
plot(ts, [1 .- norm(sol(t)) for t in ts], label="norm error", xlabel="t")

##
parities = [(1, 4), (2, 5), (3, 4)]
parities = [(0, 1), (2, 4), (3, 5), (1, 4)]
measurements = map(p -> P[p...], parities)
plot(plot(ts, [real(sol(t)'m * sol(t)) for m in measurements, t in ts]', label=permutedims(parities), legend=true, xlabel="t", ylims=(-1, 1)),
    plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=1), layout=(2, 1), lw=2, frame=:box)

##
U0 = Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1))
use_static_arrays && (U0 = SMatrix{2^(N - 1),2^(N - 1)}(U0))
prob_full = ODEProblem{!use_static_arrays}(M, U0, tspan, p)
@time sol_full = solve(prob_full, Tsit5(), saveat=ts, reltol=1e-12, tstops=ts);
sol_full(2T)
##
# Do a sweep over several zetas, solve the system for the final time t=2T and measure the parities
zetas = range(0, 1, length=100)
parities_arr = zeros(ComplexF64, length(zetas), length(measurements))
correction = 1

@showprogress for (idx, ζ) in enumerate(zetas)
    p = (T, Δmin, Δmax, k, ϵs, (ζ, ζ, ζ), correction, P)
    prob = ODEProblem{!use_static_arrays}(M, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts)
    parities_arr[idx, :] = [real(sol(2T)'m * sol(2T)) for m in measurements]
    #println("ζ = $ζ, parities = $(parities_arr[idx, :])")
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
k = 5e2

gridpoints = 50
T_arr = range(1e2, 3e3, length=gridpoints) * 1 / Δmax
zetas = range(0, 1, length=gridpoints)
parities_after_T_2D = zeros(ComplexF64, gridpoints, gridpoints, length(measurements))
parities_arr_2D = zeros(ComplexF64, gridpoints, gridpoints, length(measurements))
correction = 1

using Base.Threads

@time @showprogress for (idx_T, T) in enumerate(T_arr)
    # Please write the above loop over zetas as parallelized loop below this Line
    Threads.@threads for idx_z in 1:gridpoints
        tspan = (0.0, 2T)
        ζ = zetas[idx_z]
        ts = range(0, tspan[2], 10)
        p = (T, Δmin, Δmax, k, ϵs, (ζ, ζ, ζ), correction, P)
        prob = ODEProblem{!use_static_arrays}(M, u0, tspan, p)
        sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts)
        parities_after_T_2D[idx_z, idx_T, :] = [real(sol(T)'m * sol(T)) for m in measurements]
        parities_arr_2D[idx_z, idx_T, :] = [real(sol(2T)'m * sol(2T)) for m in measurements]
        #println("T = $T, ζ = $ζ, parities = $(parities_arr_2D[idx_T, idx_z, :])")
    end
end
##
# Plot the parities and parities_after_T (2, 4) as a continous colormap  
heatmap(T_arr, zetas, real(parities_arr_2D[:, :, 2]), xlabel="T", ylabel="ζ", title="Parity (2, 4)", c=:viridis)
##
# Change evaluation of the succesfull braiding protocol from one initial state to classifying the operator 
# that is applied to any initial state
# Define as initial state the identity Matrix
U0 = Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1))
use_static_arrays && (U0 = SMatrix{2^(N - 1),2^(N - 1)}(U0))

# As comparison to the perfect outcome we will use the evolution for the perfect fine tuned case
Δmax = 1
Δmin = 0 * Δmax
T = 2e3 / Δmax
k = 2e1

p = (T, Δmin, Δmax, k, ϵs, ζs, correction, P)
prob_full = ODEProblem{!use_static_arrays}(M, U0, tspan, p)
ts = range(0, tspan[2], 1000)
@time sol_full = solve(prob_full, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts);
perfect_operator = sol_full(2T)
println("Perfect operator = $perfect_operator")
##
# Define operator norm to classify other operators with respect to the perfect operator
function operator_norm(A, B)
    return norm(A - B)
end
##
# Do a sweep over several zetas, solve the system for the final time t=2T and measure the operator norm
zetas = range(0, 1, length=100)
operator_norm_arr = zeros(Float64, length(zetas))
correction = 1

@showprogress for (idx, ζ) in enumerate(zetas)
    p = (T, Δmin, Δmax, k, ϵs, (ζ, ζ, ζ), correction, P)
    prob = ODEProblem{!use_static_arrays}(M, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts)
    operator_norm_arr[idx] = operator_norm(sol(2T), perfect_operator)
    #println("ζ = $ζ, operator norm = $(operator_norm_arr[idx])")
end
##
# Plot the operator norm as a function of the zetas
plot(zetas, operator_norm_arr, label="Operator norm", xlabel="ζ", ylabel="Operator norm", lw=2, frame=:box)
##