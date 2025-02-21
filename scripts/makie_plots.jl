using MajoranaBraiding
using QuantumDots
using Majoranas
using LinearAlgebra
using CairoMakie
using OrdinaryDiffEq
using ProgressMeter
using StaticArrays
using Base.Threads
using Accessors
using UnPack
##
γ = get_majorana_basis()
totalparity = 1
use_static_arrays = true
inplace = !use_static_arrays
mtype, vtype = MajoranaBraiding.matrix_vec_types(use_static_arrays, inplace, 3)
P = parity_operators(γ, totalparity, mtype)
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[:M, :M̃] + P[:L, :L̃] + P[:R, :R̃]), 1:1).vectors))))
U0 = mtype(Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)))

## plot spectrum
function plot_spectrum(dict::Dict; kwargs...)
    @unpack H, p, ts, T = dict
    spectrum = stack([eigvals(H(p, t)) for t in ts])'
    plot(ts / T, mapslices(v -> v[2:end] .- v[1], spectrum, dims=2), xlabel="t/T", ls=[:solid :dash :dot], ylabel="Eᵢ-E₀", labels=[1, 2, 3]', yscale=:log10, ylims=(1e-16, 1e1), lw=2, frame=:box, kwargs...)
end

## Do a sweep over several zetas, solve the system for the final time t=2T and measure the parities
zetas = range(0, 1, length=50)
parity_measurements = MajoranaBraiding.default_parity_pairs
parity_labels = MajoranaBraiding.parity_labels(parity_measurements)
parities_arr = zeros(ComplexF64, length(zetas), length(parity_measurements))
@time @showprogress @threads for (idx, ζ) in collect(enumerate(zetas))
    local_dict = Dict(
        :ζ => ζ,
        :ϵs => (0, 0, 0),
        :T => 2e3,
        :Δmax => 1 * [1 / 3, 1 / 2, 1],
        :Δmin => 1e-6 * [2, 1 / 3, 1],
        :k => 1e1,
        :steps => 2000,
        :correction => InterpolatedExactSimpleCorrection(),
        :interpolate_corrected_hamiltonian => false,
        :P => P,

        :γ => γ,
        :u0 => u0
    )
    prob = setup_problem(local_dict)
    sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6)
    parities_arr[idx, :] = measure_parities(sol(2prob[:T]), prob, parity_measurements)
end
## Make a plot with makie
f = Figure()
ax = Axis(f[1, 1],
    xlabel="ζ",
    ylabel="Parity")
for (i, ps) in enumerate(eachcol(parities_arr))
    lines!(ax, zetas, real(ps), label=parity_labels[i])
end
axislegend()
f

## Do a sweep over the total braiding time T and the zetas and plot the parities
gridpoints = 20
T_arr = range(1e2, 1e3, length=gridpoints)
zetas = range(0, 1, length=gridpoints)
parities_after_T_2D = zeros(ComplexF64, gridpoints, gridpoints, length(parity_measurements))
parities_arr_2D = zeros(ComplexF64, gridpoints, gridpoints, length(parity_measurements))

@time @showprogress for (idx_T, T) in enumerate(T_arr)
    Threads.@threads for idx_z in 1:gridpoints
        local_dict = Dict(
            :ζ => zetas[idx_z],
            :ϵs => (0, 0, 0),
            :T => T,
            :Δmax => 1 * [1 / 3, 1 / 2, 1],
            :Δmin => 1e-6 * [2, 1 / 3, 1],
            :k => 1e1,
            :steps => 2000,
            :correction => InterpolatedExactSimpleCorrection(),
            :interpolate_corrected_hamiltonian => false,
            :P => P,
    
            :γ => γ,
            :u0 => u0
        )
        prob = setup_problem(local_dict)
        sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6)
        parities_after_T_2D[idx_z, idx_T, :] = measure_parities(sol(prob[:T]), prob, parity_measurements)
        parities_arr_2D[idx_z, idx_T, :] = measure_parities(sol(2prob[:T]), prob, parity_measurements)
    end
end
let n = 3
    heatmap(T_arr, zetas, real(parities_arr_2D[:, :, n])', axis=(; xlabel="T", ylabel="ζ", title="Parity $(parity_measurements[n])"), colorrange=(-1, 1), colormap=:viridis) |> display
end

## Calculate full solution for T and 2T and calculate the fidelities
gridpoints = 10
Ts = range(1e2, 3e3, length=gridpoints)
zetas = range(1e-3, 1 - 1e-3, length=3 * gridpoints)
single_braid_fidelity = zeros(length(Ts), length(zetas))
double_braid_fidelity = zero(single_braid_fidelity)
@time @showprogress for (idx_T, T) in enumerate(Ts)
    Threads.@threads for (idx_z, ζ) in collect(enumerate(zetas))
        local_dict = Dict(
            :ζ => ζ,
            :ϵs => (0, 0, 0),
            :T => T,
            :Δmax => 1 * [1 / 3, 1 / 2, 1],
            :Δmin => 1e-6 * [2, 1 / 3, 1],
            :k => 1e1,
            :steps => 2000,
            :correction => InterpolatedExactSimpleCorrection(),
            :interpolate_corrected_hamiltonian => false,
            :P => P,
    
            :γ => γ,
            :u0 => U0
        )
        prob = setup_problem(local_dict)
        sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6, saveat=[0, T, 2T])
        proj = Diagonal([0, 1, 1, 0])
        # proj = Diagonal([1, 0, 0, 1])
        single_braid_gate = majorana_exchange(-P[:L, :R])
        # single_braid_gate = analytical_protocol_gate(prob.dict)
        double_braid_gate = single_braid_gate^2
        single_braid_result = sol(T)
        double_braid_result = sol(2T)
        single_braid_fidelity[idx_T, idx_z] = gate_fidelity(proj * single_braid_gate * proj, proj * single_braid_result * proj)
        double_braid_fidelity[idx_T, idx_z] = gate_fidelity(proj * double_braid_gate * proj, proj * double_braid_result * proj)
    end
end
## Makie.jl plots with heatmaps in a grid
f = Figure(; size=(800, 400))
ax1 = Axis(f[1, 1], xlabel="T", ylabel="ζ", title="Single braid fidelity")
ax2 = Axis(f[1, 2], xlabel="T", ylabel="ζ", title="Double braid fidelity")
hm1 = heatmap!(ax1, T_arr, zetas, single_braid_fidelity .^ 2, colorrange=(0, 1))
hm2 = heatmap!(ax2, T_arr, zetas, double_braid_fidelity .^ 2, colorrange=(0, 1))
f

# plot(heatmap(T_arr, zetas, single_braid_fidelity .^ 2, xlabel="T", ylabel="ζ", c=:viridis, title="Single braid fidelity", clim=(0, 1)),
# heatmap(T_arr, zetas, double_braid_fidelity .^ 2, xlabel="T", ylabel="ζ", c=:viridis, title="Double braid fidelity", clim=(0, 1)))

## 1d sweep over zeta for the fidelity
gridpoints = 40
zetas = range(0, 1, length=gridpoints)
single_braid_ideal_fidelity = zeros(Float64, gridpoints)
single_braid_kato_fidelity = zeros(Float64, gridpoints)
double_braid_ideal_fidelity = zeros(Float64, gridpoints)
double_braid_kato_fidelity = zeros(Float64, gridpoints)
angles = zeros(Float64, gridpoints)
analytical_angle = zeros(Float64, gridpoints)
parity_measurements = MajoranaBraiding.default_parity_pairs
parity_labels = MajoranaBraiding.parity_labels(parity_measurements)
parities = zeros(Float64, length(zetas), length(parity_measurements))

ideal_single_braid = majorana_exchange(-P[:L, :R])
fermion_single_braid = majorana_exchange(-P[:L, :R]) * majorana_exchange(-P[:L̃, :R̃])
gate_labels = ["ideal majorana", "kato", "ideal fermion", "fit", "lucky"]
single_fidelities = zeros(Float64, gridpoints, length(gate_labels))
double_fidelities = zeros(Float64, gridpoints, length(gate_labels))
analytical_gates_fidelities = zeros(Float64, gridpoints)
@time @showprogress @threads for (n, ζ) in collect(enumerate(zetas))
    local_dict = Dict(
        :ζ => ζ,
        :ϵs => (0, 0, 0),
        :T => 4e4,
        :Δmax => 1 * [1 / 3, 1 / 2, 1],
        :Δmin => 1e-10 * [2, 1 / 3, 1],
        :k => 5e1,
        :steps => 50,
        :correction => InterpolatedExactSimpleCorrection(),
        # :correction => EigenEnergyCorrection(),
        # :correction => NoCorrection(),
        # :correction => SimpleCorrection(),
        :interpolate_corrected_hamiltonian => true,
        :P => P,

        :γ => γ,
        :u0 => U0,
        # :opt_kwargs => (; xatol=0, xrtol=0, atol=0, rtol=0, verbose=false)
        :extra_shifts => [-0.05, 0, 0]
    )
    T = local_dict[:T]
    prob = setup_problem(local_dict)
    if n == 1
        visualize_deltas(prob) |> display
    end
    #Tsit5()
    sol = solve(prob[:odeprob], Vern7(), abstol=1e-10, reltol=1e-10, saveat=[0, T, 2T])
    single_braid_result = sol(T)
    double_braid_result = sol(2T)

    proj = totalparity == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
    angles[n] = braid_gate_best_angle(single_braid_result, P)[1]
    single_braid_gates = [ideal_single_braid, MajoranaBraiding.single_braid_gate_kato(prob), fermion_single_braid, MajoranaBraiding.single_braid_gate_fit(angles[n], P), MajoranaBraiding.single_braid_gate_lucky_guess(prob)]
    double_braid_gates = [g^2 for g in single_braid_gates]

    # analytical_angle[n] = MajoranaBraiding.zero_energy_analytic_parameters(prob) #single_braid_gate_analytical_angle(prob)
    # fidelities[n] = braid_gate_best_angle(single_braid_result, P)[2]
    single_fidelities[n, :] .= [gate_fidelity(proj * target_gate * proj, proj * single_braid_result * proj) for target_gate in single_braid_gates]
    double_fidelities[n, :] .= [gate_fidelity(proj * target_gate * proj, proj * double_braid_result * proj) for target_gate in double_braid_gates]
    parities[n, :] = measure_parities(sol(2T) * u0, prob, parity_measurements)
    analytical_gates_fidelities[n] = gate_fidelity(proj * double_braid_gates[2] * proj, proj * double_braid_gates[5] * proj)
    # norm(I - double_braid_gates[2]' * double_braid_gates[2]) |> println
    # norm(I - double_braid_gates[5]' * double_braid_gates[5]) |> println
    norm(proj - (x -> x / sign(tr(x)))(proj * double_braid_gates[5]' * proj * double_braid_gates[2] * proj)) |> display
end
f, ax, pl = plot(1 .- double_fidelities[:, 2]; axis=(; yscale=log10));
ylims!(1e-12, 1e-1);
plot!(ax, 1 .- double_fidelities[:, 5]);
f
##
plot(1 .- analytical_gates_fidelities)
## Make a makie grid layout with all fidelities
let labels_fidelities = collect(zip(gate_labels, eachcol(single_fidelities), eachcol(double_fidelities)))[[2, 5]]
    f = Figure(; size=(400, 300), fontsize=15)
    ax = Axis(f[1, 1], xlabel="ζ", ylabel="Fidelity")
    ylims!(0 - 1e-3, 1 + 1e-1)
    xlims!(0 - 1e-3, 1 + 1e-3)
    linewidth = 2
    kwargs = (; linewidth)
    for (i, (label, singles, doubles)) in enumerate(labels_fidelities)
        # lines!(ax, zetas .^ 1, singles; linestyle=:dash, label, color=Cycled(i), kwargs...)
        lines!(ax, zetas .^ 1, doubles; label=label, color=Cycled(i), kwargs...)
    end

    #Make legend
    dash = LineElement(color=:black, linestyle=:dash; linewidth)
    solid = LineElement(color=:black, linestyle=:solid; linewidth)
    color1 = MarkerElement(color=Cycled(1), marker=:rect, markersize=10)
    color2 = MarkerElement(color=Cycled(2), marker=:rect, markersize=10)
    axislegend(ax,
        [dash, solid, color1, color2],
        ["Single", "Double", "Majorana", "Fermion"],
        tellwidth=false,
        position=:lb)
    axislegend(ax; position=:lt)
    f
end
## Same plot for parities 
f = Figure(; size=(400, 300))
ax = Axis(f[1, 1], xlabel="ζ", ylabel="Parity")
for (i, label) in enumerate(parity_labels)
    lines!(ax, zetas, real(parities[:, i]); label=label, color=Cycled(i), linewidth=3)
end
axislegend(; position=(:left, :center))
f


##
f = Figure()
ax = Axis(f[1, 1], xlabel="ζ", ylabel="Fidelity")
ylims!(0, 1)
for (label, fidelity) in collect(zip(gate_labels))
    lines!(ax, zetas .^ 2, fidelity; label)
end
axislegend(; position=(:left, :center))
f
## now for double braid
# f = Figure()
ax = Axis(f[2, 1], xlabel="ζ", ylabel="Fidelity")
for (i, label) in enumerate(gate_labels)
    lines!(ax, zetas .^ 2, double_fidelities[:, i], label=label)
end
axislegend(; position=(:left, :center))
f

## only kato and fit
f = Figure()
ax = Axis(f[1, 1], xlabel="ζ", ylabel="Fidelity")
for (label, data) in collect(zip(gate_labels, eachcol(single_fidelities)))[[2, 4]]
    lines!(ax, zetas .^ 2, data, label=label)
end
axislegend(; position=(:left, :center))
f
## Plot analytical angles
f = Figure()
ax = Axis(f[1, 1], xlabel="ζ", ylabel="Angle")
lines!(ax, zetas .^ 2, angles, label="best angle")
lines!(ax, zetas .^ 2, analytical_angles[:, 1], label="analytical θ")
lines!(ax, zetas .^ 2, analytical_angles[:, 2], label="analytical ϕ")
axislegend(; position=(:left, :center))
f
##
single_fidelities[:, 1] .- cos.((analytical_angle .- pi / 4)) .^ 2 |> norm
double_fidelities[:, 1] .- cos.((analytical_angle .- pi / 4) * 2) .^ 2 |> norm
parities[:, 1] .- cos.(angles) .^ 2 |> norm


## Let's look at the analytical optimization problem
params = (; Δmax=1 * [1 / 3, 100 / 2, 100], Δmin=1e-6 * [2, 1 / 3, 1], k=1e1, T=2e4)
ramp = RampProtocol(params.Δmin, params.Δmax, params.T, params.k)
xs = range(-1, 1, length=100)
zetas = range(0, 1, length=117)
energy_splittings = [MajoranaBraiding.energy_splitting(x, ζ, ramp, params.T / 2) for x in xs, ζ in zetas]
fig, ax, hm = heatmap(xs, zetas, energy_splittings, colormap=:redsblues, colorrange=(-1, 1), axis=(; xlabel="x", ylabel="ζ", title="Energy splitting"))
Colorbar(fig[:, end+1], hm)
fig
##
plot(zetas, single_braid_ideal_fidelity, label="single_braid_ideal_fidelity", xlabel="ζ", lw=2, frame=:box)
plot!(zetas, double_braid_ideal_fidelity, label="double_braid_ideal_fidelity", lw=2, frame=:box)
plot!(zetas, single_braid_kato_fidelity, label="single_braid_kato_fidelity", lw=2, frame=:box)
plot!(zetas, double_braid_kato_fidelity, label="double_braid_kato_fidelity", lw=2, frame=:box)
plot!(zetas, 1 .- (angles .- analytical_angles) .^ 2, label="1- (angles - analytical_angles)^2", xlabel="ζ", lw=2, frame=:box)
## plot angles 
plot(zetas, angles, label="angles", xlabel="ζ", lw=2, frame=:box)
plot!(zetas, analytical_angles, label="analytical_angles", lw=2, frame=:box)
##
plot(zetas, single_braid_kato_fidelity, label="single_braid_kato_fidelity", lw=2, frame=:box)
plot!(zetas, double_braid_kato_fidelity, label="double_braid_kato_fidelity", lw=2, frame=:box)
