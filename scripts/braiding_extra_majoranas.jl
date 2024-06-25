# Braiding following Beenakker's review 1907.06497
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
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[0, 1] + P[2, 4] + P[3, 5]), 1:1).vectors))))
Δmax = 1
T = 1e3 / Δmax
k = 1e1
Δmin = 1e-6 * Δmax
ϵs = (0.0, 0.0, 0.0) # Energy overlaps between Majoranas ordered as ϵ01, ϵ24, ϵ35
ζ = 1e-2
ζs = (ζ, ζ, ζ) # Unwanted Majorana contributions within each island ordered as ζ01, ζ24, ζ35
correction = 0
tspan = (0.0, 2T)
ramp = RampProtocol([2, 1/3, 1] .* Δmin, [1 / 3, 1 / 2, 1] .* Δmax, T, k)
p = (ramp, ϵs, ζs, correction, P)
H = ham_with_corrections
H! = ham_with_corrections!
M = get_op(H, H!, p)

##
prob = ODEProblem{inplace}(M, u0, tspan, p)
ts = range(0, tspan[2], 1000)
deltas = stack([ramp(t) for t in ts])'
delta_plot = plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)
spectrum = stack([eigvals(H(p, t)) for t in ts])'
plot(plot(mapslices(v -> v[2:end] .- v[1], spectrum, dims=2), ls=[:solid :dash :dot], title="Eᵢ-E₀", labels=[1, 2, 3]', yscale=:log10), delta_plot, layout=(2, 1), lw=2, frame=:box)
## Solve the system
@time sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts);
plot(ts, [1 .- norm(sol(t)) for t in ts], label="norm error", xlabel="t")

##
parities = [(0, 1), (2, 4), (3, 5)] # (1, 4), (2, 5)
measurements = map(p -> P[p...], parities)
expval(m::AbstractMatrix, ψ) = dot(ψ, m, ψ)
plot(plot(ts, [real(expval(m, sol(t))) for m in measurements, t in ts]', label=permutedims(parities), legend=true, xlabel="t", ylims=(-1, 1)),
    delta_plot, layout=(2, 1), lw=2, frame=:box)

##
U0 = mtype(Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)))
prob_full = ODEProblem{inplace}(M, U0, tspan, p)
@time sol_full = solve(prob_full, Tsit5(), saveat=ts, reltol=1e-6, abstol=1e-6, tstops=ts);
sol_full(2T)
##
# Do a sweep over several zetas, solve the system for the final time t=2T and measure the parities
zetas = range(0, 1, length=500)
parities_arr = zeros(ComplexF64, length(zetas), length(measurements))
correction = 1

@time @showprogress @threads for (idx, ζ) in collect(enumerate(zetas))
    p = (ramp, ϵs, (ζ, ζ, ζ), correction, P)
    prob = ODEProblem{inplace}(M, u0, tspan, p)
    ts = [0, T, 2T]
    sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts)
    parities_arr[idx, :] = [real(expval(m, sol(2T))) for m in measurements]
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
k = 1e1

gridpoints = 40
T_arr = range(1e2, 3e3, length=gridpoints) * 1 / Δmax
zetas = range(0, 1, length=gridpoints)
parities_after_T_2D = zeros(ComplexF64, gridpoints, gridpoints, length(measurements))
parities_arr_2D = zeros(ComplexF64, gridpoints, gridpoints, length(measurements))
correction = 1

@time @showprogress for (idx_T, T) in enumerate(T_arr)
    # Please write the above loop over zetas as parallelized loop below this Line
    Threads.@threads for idx_z in 1:gridpoints
        tspan = (0.0, 2T)
        ζ = zetas[idx_z]
        ts = tstops = [0, T, 2T]#range(0, tspan[2], 1000)
        ramp = RampProtocol([1, 1, 1] .* Δmin, [1 / 3, 1 / 2, 1] .* Δmax, T, k)
        p = (ramp, ϵs, (ζ, ζ, ζ), correction, P)
        prob = ODEProblem{inplace}(M, u0, tspan, p)
        sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-6, tstops=ts, saveat=ts)
        parities_after_T_2D[idx_z, idx_T, :] = [real(expval(m, sol(T))) for m in measurements]
        parities_arr_2D[idx_z, idx_T, :] = [real(expval(m, sol(2T))) for m in measurements]
        #println("T = $T, ζ = $ζ, parities = $(parities_arr_2D[idx_T, idx_z, :])")
    end
end
##
# Plot the parities and parities_after_T (2, 4) as a continous colormap  
heatmap(T_arr, zetas, abs.(parities_arr_2D[:, :, 3]), xlabel="T", ylabel="ζ", c=:viridis, title="Parity (0, 1)", clim=(-1, 1)) |> display
##
