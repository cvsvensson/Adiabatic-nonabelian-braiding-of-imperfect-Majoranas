# Braiding following Beenakker's review 1907.06497
using MajoranaBraiding
using QuantumDots
using Majoranas
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using ProgressMeter
using StaticArrays
using Base.Threads
# using TaylorSeries
using Roots

##
nbr_of_majoranas = 6
N = nbr_of_majoranas ÷ 2
majorana_labels = 0:5
γ = SingleParticleMajoranaBasis(nbr_of_majoranas, majorana_labels)
parity = 1
use_static_arrays = true
inplace = !use_static_arrays
mtype, vtype = MajoranaBraiding.matrix_vec_types(use_static_arrays, inplace, N)
P = parity_operators(γ, parity, mtype)
#=Pold = MajoranaBraiding.parity_operators_old(nbr_of_majoranas, majorana_labels, mtype)=#

H = ham_with_corrections
P
## Parameters
P
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[0, 1] + P[2, 4] + P[3, 5]), 1:1).vectors))))
Δmax = 1
T = 1e3 / Δmax
k = 1e1
Δmin = 1e-6 * Δmax
ϵs = (0.0, 0.0, 0.0) # Energy overlaps between Majoranas ordered as ϵ01, ϵ24, ϵ35
ζ = 9e-1
ζs = (ζ, ζ, ζ) # Unwanted Majorana contributions within each island ordered as ζ01, ζ24, ζ35
tspan = (0.0, 2T)
# Take ts with one step per time unit
dt = 2
ts = range(0, tspan[2], Int(tspan[2] / dt))
ramp = RampProtocol([2, 1, 1] .* Δmin, [1, 1, 1] .* Δmax, T, k)
simplecorr = optimized_simple_correction(H, (ramp, ϵs, ζs, P), ts)
independentsimplecorr = optimized_independent_simple_correction(H, (ramp, ϵs, ζs, P), ts)
analyticsimplecorr = analytical_exact_simple_correction(ζ, ramp, ts)
remove_labels = [[0,1], [0,2], [0,3]]
constrained_basis = MajoranaBraiding.remove_from_basis(remove_labels, P)
#=constrained_basis = P=#
eigencorrection = EigenEnergyCorrection(constrained_basis)
##
corr = eigencorrection
p = (ramp, ϵs, ζs, corr, P)
M = get_op(H, p)
## Solve the system
prob = ODEProblem{inplace}(M, u0, tspan, p)
@time sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6, tstops=ts);
plot(ts, [1 .- norm(sol(t)) for t in ts], label="norm error", xlabel="t")
##
deltas = stack([ramp(t) for t in ts])'
delta_plot = plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)
spectrum = stack([eigvals(H(p, t)) for t in ts])'
plot(plot(ts, mapslices(v -> v[2:end] .- v[1], spectrum, dims=2), ls=[:solid :dash :dot], title="Eᵢ-E₀", labels=[1, 2, 3]', yscale=:log10, ylims=(1e-16, 1e1)), delta_plot, layout=(2, 1), lw=2, frame=:box)

##
parities = [(0, 1), (2, 4), (3, 5)] #, (1, 4), (2, 5)
measurements = map(p -> P[p...], parities)
expval(m::AbstractMatrix, ψ) = dot(ψ, m, ψ)
plot(plot(ts, [real(expval(m, sol(t))) for m in measurements, t in ts]', label=permutedims(parities), legend=true, xlabel="t", ylims=(-1, 1)),
    delta_plot, layout=(2, 1), lw=2, frame=:box)

##
U0 = mtype(Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)))
prob_full = ODEProblem{inplace}(M, U0, tspan, p)
@time sol_full = solve(prob_full, Tsit5(), saveat=ts, reltol=1e-6, abstol=1e-6, tstops=ts);
single_braid_gate = majorana_exchange(-P[2, 3])
single_braid_gate = single_braid_gate_improved(P, ζ, ramp, T)
double_braid_gate = single_braid_gate^2
single_braid_result = sol_full(T)
double_braid_result = sol_full(2T)
proj = Diagonal([0, 1, 1, 0])
single_braid_fidelity = gate_fidelity(proj * single_braid_gate * proj, proj * single_braid_result * proj)
println("Single braid fidelity: ", single_braid_fidelity)
double_braid_fidelity = gate_fidelity(proj * double_braid_gate * proj, proj * double_braid_result * proj)
println("Double braid fidelity: ", double_braid_fidelity)
##
# Do a sweep over several zetas, solve the system for the final time t=2T and measure the parities
zetas = range(1e-6, 1 - 1e-2, length=20)
parities_arr = zeros(ComplexF64, length(zetas), length(measurements))
T = 1e4
tspan = (0.0, 2T)
ts = range(0, tspan[2], 1000)
@time @showprogress @threads for (idx, ζ) in collect(enumerate(zetas))
    ramp = RampProtocol([2, 1 / 3, 1] .* Δmin, [1 / 3, 1 / 2, 1] .* Δmax, T, k)
    ζs = (ζ, ζ,  ζ / 1)
    corr = analytical_exact_simple_correction(ζ, ramp, ts)
    #corr = optimized_independent_simple_correction(H, (ramp, ϵs, ζs, P), ts)
    #corr = SimpleCorrection()
    p = (ramp, ϵs, ζs, corr, P)
    prob = ODEProblem{inplace}(M, u0, tspan, p)
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
Δmin = 1e-6 * Δmax
ϵs = Δmax * [0.0, 0.0, 0.0]
k = 1e1

gridpoints = 5
T_arr = range(1e2, 3e3, length=gridpoints) * 1 / Δmax
zetas = range(0, 1, length=gridpoints)
parities_after_T_2D = zeros(ComplexF64, gridpoints, gridpoints, length(measurements))
parities_arr_2D = zeros(ComplexF64, gridpoints, gridpoints, length(measurements))

@time @showprogress for (idx_T, T) in enumerate(T_arr)
    # Please write the above loop over zetas as parallelized loop below this Line
    Threads.@threads for idx_z in 1:gridpoints
        tspan = (0.0, 2T)
        ζ = zetas[idx_z]
        ts = range(0, tspan[2], 1000)
        ramp = RampProtocol([1, 1, 1] .* Δmin, [1 / 3, 1 / 2, 1] .* Δmax, T, k)
        #corr = optimized_simple_correction(H, (ramp, ϵs, (ζ, ζ, ζ), P), ts)
        corr = analytical_exact_simple_correction(ζ, ramp, ts)
        # corr = SimpleCorrection()
        p = (ramp, ϵs, (ζ, ζ, ζ), corr, P)
        prob = ODEProblem{inplace}(M, u0, tspan, p)
        sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-6)
        parities_after_T_2D[idx_z, idx_T, :] = [real(expval(m, sol(T))) for m in measurements]
        parities_arr_2D[idx_z, idx_T, :] = [real(expval(m, sol(2T))) for m in measurements]
        #println("T = $T, ζ = $ζ, parities = $(parities_arr_2D[idx_T, idx_z, :])")
    end
end
##
# Plot the parities and parities_after_T (2, 4) as a continous colormap  
let n = 3
    heatmap(T_arr, zetas, real(parities_arr_2D[:, :, n]), xlabel="T", ylabel="ζ", c=:viridis, title="Parity $(parities[n])", clim=(-1, 1)) |> display
end

## Calculate full solution for T and 2T and calculate the fidelities
gridpoints = 5
T_arr = range(1e2, 3e3, length=gridpoints) * 1 / Δmax
zetas = range(1e-3, 1-1e-3, length=3 * gridpoints)
single_braid_fidelity = zeros(Float64, 3gridpoints, gridpoints)
double_braid_fidelity = zeros(Float64, 3gridpoints, gridpoints)
@time @showprogress for (idx_T, T) in enumerate(T_arr)
    Threads.@threads for (idx_z, _z) in collect(enumerate(zetas))
        tspan = (0.0, 2T)
        ζ = zetas[idx_z]
        ts = range(0, tspan[2], 1000)
        ramp = RampProtocol([1, 1, 1] .* Δmin, [1 / 3, 1 / 2, 1] .* Δmax, T, k)
        correction = analytical_exact_simple_correction(ζ, ramp, ts)
        #correction = SimpleCorrection()
        p = (ramp, ϵs, (ζ, ζ, ζ), correction, P)
        prob = ODEProblem{inplace}(M, U0, tspan, p)
        sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-6, saveat=[0, T, 2T])
        proj = Diagonal([0, 1, 1, 0])
        # proj = Diagonal([1, 0, 0, 1])
        single_braid_gate = majorana_exchange(-P[2, 3])
        single_braid_gate = single_braid_gate_improved(P, ζ, ramp, T)
        double_braid_gate = single_braid_gate^2
        single_braid_result = sol(T)
        double_braid_result = sol(2T)
        single_braid_fidelity[idx_z, idx_T] = gate_fidelity(proj * single_braid_gate * proj, proj * single_braid_result * proj)
        double_braid_fidelity[idx_z, idx_T] = gate_fidelity(proj * double_braid_gate * proj, proj * double_braid_result * proj)
    end
end
##
plot(heatmap(T_arr, zetas, single_braid_fidelity .^ 2, xlabel="T", ylabel="ζ", c=:viridis, title="Single braid fidelity", clim=(0, 1)),
    heatmap(T_arr, zetas, double_braid_fidelity .^ 2, xlabel="T", ylabel="ζ", c=:viridis, title="Double braid fidelity", clim=(0, 1)))


## 1d sweep over zeta for the fidelity
gridpoints = 5
zetas = range(1e-3, 1-1e-3, length=gridpoints)
# logrange(x, y, n) = exp10.(range(log10(x), log10(y), length=n))
# zetas = logrange(1e-4, 1, gridpoints)
single_braid_fidelity = zeros(Float64, gridpoints)
double_braid_fidelity = zeros(Float64, gridpoints)
ts = range(0, tspan[2], 2000)
@time @showprogress @threads for (idx, ζ) in collect(enumerate(zetas))
    ramp = RampProtocol([1, 1 / 2, 1 / 4] .* 0, [1 / 3, 1 / 2, 1] .* Δmax, T, k)
    # corr = optimized_simple_correction(H, (ramp, ϵs, (ζ, ζ, ζ), P), ts)
    corr = analytical_exact_simple_correction(ζ, ramp, ts)
    p = (ramp, ϵs, (ζ, ζ, ζ), corr, P)
    prob = ODEProblem{inplace}(M, U0, tspan, p)
    sol = solve(prob, Tsit5(), abstol=1e-9, reltol=1e-9, saveat=[0, T, 2T])
    proj = Diagonal([0, 1, 1, 0])
    single_braid_gate = majorana_exchange(-P[2, 3])
    single_braid_gate = single_braid_gate_improved(P, ζ, ramp, T)
    double_braid_gate = single_braid_gate^2
    single_braid_result = sol(T)
    double_braid_result = sol(2T)
    single_braid_fidelity[idx] = gate_fidelity(proj * single_braid_gate * proj, proj * single_braid_result * proj)
    double_braid_fidelity[idx] = gate_fidelity(proj * double_braid_gate * proj, proj * double_braid_result * proj)
end
##
# Plot the parities as a function of the zetas
plot(zetas, single_braid_fidelity, label="single_braid_fidelity", xlabel="ζ", lw=2, frame=:box);
plot!(zetas, double_braid_fidelity, label="double_braid_fidelity", lw=2, frame=:box)
##
plt = plot(frame=:box)
plot!(plt, zetas, 1 .- single_braid_fidelity, label="1 - F1", xlabel="ζ", lw=2, yscale=:log10, xscale=:log10, ylims=(1e-16, 1), markers=true, leg=:topleft);
plot!(plt, zetas, 1 .- double_braid_fidelity, label="1 - F2", lw=2, markers=true, yscale=:log10, xscale=:log10)
vline!(plt, [0.5], lw=1, c=:black, ls=:dashdot, label="ζ=0.5")

twinplt = twinx()
plot!(twinplt, zetas[2:end], diff(log.(1 .- double_braid_fidelity)) ./ diff(log.(zetas)), ylims=(0, 9), xscale=:log10, label="∂log(1 - F2)/∂log(ζ)", lw=2, yticks=10, markers=true, grid=false, c=3, legend=:bottomright)
hline!(twinplt, [4, 8], lw=1, c=:black, ls=:dash, label="slope = [4, 8]")
##
# plot(plt, plot((zetas[2:end]), diff(log.(1 .- double_braid_fidelity)) ./ diff(log.(zetas)), ylims=(0, 9), xscale=:log10, label="∂log(1 - F)/∂log(ζ)", lw=2, yticks=10, frame=:box, markers=true))
##
plot(range(0.1, 1, length=gridpoints), range(0.1, 1, length=gridpoints), label=false, yaxis=:log10)
##
let xscale = :identity, zetas = zetas
    plt = plot(frame=:box)
    plot!(plt, zetas, 1 .- single_braid_fidelity; label="1 - F1", xlabel="ζ", lw=2, yscale=:log10, xscale, ylims=(1e-16, 1), markers=true, leg=:topleft)
    plot!(plt, zetas, 1 .- double_braid_fidelity; label="1 - F2", lw=2, markers=true, yscale=:log10, xscale)
    vline!(plt, [0.5], lw=1, c=:black, ls=:dashdot, label="ζ=0.5")

    twinplt = twinx()
    plot!(twinplt, zetas[2:end], diff(log.(1 .- double_braid_fidelity)) ./ diff(log.(zetas)); ylims=(0, 9), xscale, label="∂log(1 - F2)/∂log(ζ)", lw=2, yticks=10, markers=true, grid=false, c=3, legend=:bottomright)
    hline!(twinplt, [4, 8], lw=1, c=:black, ls=:dash, label="slope = [4, 8]")
    display(plt)
end
