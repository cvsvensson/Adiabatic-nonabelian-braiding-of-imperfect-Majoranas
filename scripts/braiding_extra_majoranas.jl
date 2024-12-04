using MajoranaBraiding
using QuantumDots
using Majoranas
using LinearAlgebra
using Plots
using OrdinaryDiffEqTsit5
using ProgressMeter
using StaticArrays
using Base.Threads
using Accessors
##
γ = get_majorana_basis()
N = length(γ.fermion_basis)
totalparity = 1
use_static_arrays = true
inplace = !use_static_arrays
mtype, vtype = MajoranaBraiding.matrix_vec_types(use_static_arrays, inplace, N)
P = parity_operators(γ, totalparity, mtype)
# H = ham_with_corrections
## Initial state and identity matrix
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[:M, :M̃] + P[:L, :L̃] + P[:R, :R̃]), 1:1).vectors))))
U0 = mtype(Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)))

##
param_dict = Dict(
    :ζ => 0.7, #Majorana overlaps. Number or triplet of numbers
    :ϵs => (0, 0, 0), #Dot energy levels
    :T => 1e4, #Maximum time
    :Δmax => 1 * (rand(3) .+ 0.5), #Largest values of Δs. Number or triplet of numbers
    :Δmin => 1e-10 * (rand(3) .+ 0.5), #Smallest values of Δs. Number or triplet of numbers
    :k => 1e1, #Determines the slope of the ramp
    :steps => 2000, #Number of timesteps for interpolations
    :correction => InterpolatedExactSimpleCorrection(), #Different corrections are available. This is the most relevant one for the paper
    :interpolate_corrected_hamiltonian => false, #Creating an interpolated Hamiltonian might speed things up
    :P => P, #Dict with parity operators
    :inplace => inplace, #Does the hamiltonian act in place? I'm not sure this works anymore
    :γ => γ, #Majorana basis
    :u0 => u0, #Initial state. Use U0 for the identity matrix.
    :extra_shifts => [0, 0, 0] #Shifts the three Δ pulses. Given as fractions of T
)

## Solve the system
prob = setup_problem(param_dict)
@time sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6);
plot(sol.t, [1 .- norm(sol(t)) for t in sol.t], label="norm error", xlabel="t")
##
visualize_spectrum(prob)
visualize_deltas(prob)
visualize_parities(sol, prob)
visualize_analytic_parameters(prob)
visualize_protocol(prob)
##
full_gate_param_dict = @set param_dict[:u0] = U0
prob_full = setup_problem(full_gate_param_dict)
@time sol_full = solve(prob_full[:odeprob], Tsit5(), reltol=1e-6, abstol=1e-6);
single_braid_gate = majorana_exchange(-P[:L, :R])
single_braid_gate = single_braid_gate_kato(prob_full)
double_braid_gate = single_braid_gate^2
single_braid_result = sol_full(prob_full[:T])
double_braid_result = sol_full(2prob_full[:T])
proj = Diagonal([0, 1, 1, 0])
single_braid_fidelity = gate_fidelity(proj * single_braid_gate * proj, proj * single_braid_result * proj)
println("Single braid fidelity: ", single_braid_fidelity)
double_braid_fidelity = gate_fidelity(proj * double_braid_gate * proj, proj * double_braid_result * proj)
println("Double braid fidelity: ", double_braid_fidelity)

println("Fit of angle for braid gate: ", braid_gate_best_angle(single_braid_gate, P))

braid_gate_prediction(single_braid_gate, single_braid_gate_analytical_angle(prob_full), P)
##
# Do a sweep over several zetas, solve the system for the final time t=2T and measure the parities
omegas = range(0, pi / 2, 50) #range(0, 2, length=50)
parity_measurements = [(:L, :L̃), (:M, :M̃), (:R, :R̃)]
parity_labels = MajoranaBraiding.parity_labels(parity_measurements)
parities_arr = zeros(ComplexF64, length(omegas), length(parity_measurements))
@time @showprogress @threads for (idx, omega) in collect(enumerate(omegas))
    local_dict = Dict(
        :ζ => tan(omega),
        :ϵs => (0, 0, 0),
        :T => 2e3,
        :Δmax => 1 * [1 / 3, 1 / 2, 1],
        :Δmin => 1e-6 * [2, 1 / 3, 1],
        :k => 1e1,
        :steps => 2000,
        :correction => InterpolatedExactSimpleCorrection(),
        :interpolate_corrected_hamiltonian => false,
        :P => P,
        :inplace => inplace,
        :γ => γ,
        :u0 => u0
    )
    prob = setup_problem(local_dict)
    sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6)
    parities_arr[idx, :] = measure_parities(sol(2prob[:T]), prob, parity_measurements)
end
plot(omegas, real(parities_arr), label=parity_labels, xlabel="ζ", ylabel="Parity", lw=2, frame=:box)
## Do a sweep over the total braiding time T and the zetas and plot the parities
gridpoints = 10
T_arr = range(1e2, 3e3, length=gridpoints)
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
            :inplace => inplace,
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
    heatmap(T_arr, zetas, real(parities_arr_2D[:, :, n]), xlabel="T", ylabel="ζ", c=:viridis, title="Parity $(parity_measurements[n])", clim=(-1, 1)) |> display
end

## Calculate full solution for T and 2T and calculate the fidelities
gridpoints = 5
T_arr = range(1e2, 3e3, length=gridpoints)
zetas = range(1e-3, 1 - 1e-3, length=3 * gridpoints)
single_braid_fidelity = zeros(Float64, 3gridpoints, gridpoints)
double_braid_fidelity = zeros(Float64, 3gridpoints, gridpoints)
@time @showprogress for (idx_T, T) in enumerate(T_arr)
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
            :inplace => inplace,
            :γ => γ,
            :u0 => U0
        )
        prob = setup_problem(local_dict)
        sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6, saveat=[0, T, 2T])
        proj = Diagonal([0, 1, 1, 0])
        # proj = Diagonal([1, 0, 0, 1])
        single_braid_gate = majorana_exchange(-P[:L, :R])
        # single_braid_gate = analytical_protocol_gate(prob)
        double_braid_gate = single_braid_gate^2
        single_braid_result = sol(T)
        double_braid_result = sol(2T)
        single_braid_fidelity[idx_z, idx_T] = gate_fidelity(proj * single_braid_gate * proj, proj * single_braid_result * proj)
        double_braid_fidelity[idx_z, idx_T] = gate_fidelity(proj * double_braid_gate * proj, proj * double_braid_result * proj)
    end
end
plot(heatmap(T_arr, zetas, single_braid_fidelity .^ 2, xlabel="T", ylabel="ζ", c=:viridis, title="Single braid fidelity", clim=(0, 1)),
    heatmap(T_arr, zetas, double_braid_fidelity .^ 2, xlabel="T", ylabel="ζ", c=:viridis, title="Double braid fidelity", clim=(0, 1)))

## 1d sweep over zeta for the fidelity
gridpoints = 50
omegas = range(0, pi / 2 - 0.01, gridpoints) #range(0, 1, length=gridpoints)
single_braid_ideal_fidelity = zeros(Float64, gridpoints)
single_braid_lucky_fidelity = zeros(Float64, gridpoints)
single_braid_kato_fidelity = zeros(Float64, gridpoints)
double_braid_ideal_fidelity = zeros(Float64, gridpoints)
double_braid_kato_fidelity = zeros(Float64, gridpoints)
double_braid_lucky_fidelity = zeros(Float64, gridpoints)
angles = zeros(Float64, gridpoints)
analytical_angles = zeros(Float64, gridpoints)
analytical_fidelity = zeros(Float64, gridpoints)
fidelities = zeros(Float64, gridpoints)
fidelity_numerics_analytic = zeros(Float64, gridpoints)
@time @showprogress @threads for (idx, omega) in collect(enumerate(omegas))
    local_dict = Dict(
        :ζ => tan(omega),
        :ϵs => (0, 0, 0),
        :T => 3e4,
        :Δmax => 1 * [1 / 3, 1 / 2, 1],
        :Δmin => 1e-10 * [2, 1 / 3, 1],
        :k => 1e1,
        :steps => 4000,
        :correction => InterpolatedExactSimpleCorrection(),
        # :correction => EigenEnergyCorrection(),
        # :correction => NoCorrection(),
        # :correction => SimpleCorrection(),
        :interpolate_corrected_hamiltonian => true,
        :P => P,
        :inplace => inplace,
        :γ => γ,
        :u0 => U0
    )
    T = local_dict[:T]
    prob = setup_problem(local_dict)
    sol = solve(prob[:odeprob], Tsit5(), abstol=1e-8, reltol=1e-8, saveat=[0, T, 2T])
    proj = totalparity == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
    single_braid_gate_ideal = majorana_exchange(-P[:L, :R])
    single_braid_lucky_guess = single_braid_gate_lucky_guess(prob)
    single_braid_gate_kato_ = single_braid_gate_kato(prob)
    double_braid_gate_ideal = single_braid_gate_ideal^2
    double_braid_gate_kato = single_braid_gate_kato_^2
    double_braid_lucky_guess = single_braid_lucky_guess^2
    single_braid_result = sol(T)
    double_braid_result = sol(2T)
    analytical_angles[idx] = single_braid_gate_analytical_angle(prob)
    angles[idx] = braid_gate_best_angle(single_braid_result, P, proj)[1]
    fidelities[idx] = braid_gate_best_angle(single_braid_result, P, proj)[2]
    single_braid_ideal_fidelity[idx] = gate_fidelity(proj * single_braid_gate_ideal * proj, proj * single_braid_result * proj)
    single_braid_kato_fidelity[idx] = gate_fidelity(proj * single_braid_gate_kato_ * proj, proj * single_braid_result * proj)
    single_braid_lucky_fidelity[idx] = gate_fidelity(proj * single_braid_lucky_guess * proj, proj * single_braid_result * proj)
    double_braid_ideal_fidelity[idx] = gate_fidelity(proj * double_braid_gate_ideal * proj, proj * double_braid_result * proj)
    double_braid_kato_fidelity[idx] = gate_fidelity(proj * double_braid_gate_kato * proj, proj * double_braid_result * proj)
    double_braid_lucky_fidelity[idx] = gate_fidelity(proj * double_braid_lucky_guess * proj, proj * double_braid_result * proj)

    analytical_fidelity[idx] = analytical_gate_fidelity(prob)
    fidelity_numerics_analytic[idx] = gate_fidelity(proj * single_braid_gate_ideal * proj, proj * MajoranaBraiding.single_braid_gate_fit(angles[idx], P) * proj)
end
##
plot(; xlabel="ω", lw=2, frame=:box)
# plot!(omegas, single_braid_ideal_fidelity, label="single_braid_ideal_fidelity", xlabel="ω", lw=2)
# plot!(omegas, double_braid_ideal_fidelity, label="double_braid_ideal_fidelity", lw=2)
plot!(omegas, single_braid_kato_fidelity, label="single_braid_kato_fidelity", lw=2)
plot!(omegas, double_braid_kato_fidelity, label="double_braid_kato_fidelity", lw=2)
plot!(omegas, analytical_fidelity, label="analytical_fidelity", lw=2)
plot!(omegas, fidelity_numerics_analytic, label="fidelity_numerics_analytic", lw=2)
plot!(omegas, single_braid_lucky_fidelity, label="single_braid_lucky_fidelity", lw=2)
plot!(omegas, double_braid_lucky_fidelity, label="double_braid_lucky_fidelity", lw=2)
# plot!(omegas, 1 .- (angles .- analytical_angles) .^ 2, label="1- (angles - analytical_angles)^2", xlabel="ω", lw=2, frame=:box)
## plot angles 
plot(omegas, angles, label="angles", xlabel="ζ", lw=2, frame=:box)
plot!(omegas, analytical_angles, label="analytical_angles", lw=2, frame=:box)
##
plot(omegas, single_braid_kato_fidelity, label="single_braid_kato_fidelity", lw=2, frame=:box)
plot!(omegas, double_braid_kato_fidelity, label="double_braid_kato_fidelity", lw=2, frame=:box)

##
let xscale = :identity, zetas = zetas, single_braid_fidelity = single_braid_ideal_fidelity, double_braid_fidelity = double_braid_ideal_fidelity
    plt = plot(frame=:box)
    plot!(plt, zetas, 1 .- single_braid_fidelity; label="1 - F1", xlabel="ζ", lw=2, yscale=:log10, xscale, ylims=(1e-16, 1), markers=true, leg=:topleft)
    plot!(plt, zetas, 1 .- double_braid_fidelity; label="1 - F2", lw=2, markers=true, yscale=:log10, xscale)
    vline!(plt, [0.5], lw=1, c=:black, ls=:dashdot, label="ζ=0.5")

    twinplt = twinx()
    plot!(twinplt, zetas[2:end], diff(log.(1 .- double_braid_fidelity)) ./ diff(log.(zetas)); ylims=(0, 9), xscale, label="∂log(1 - F2)/∂log(ζ)", lw=2, yticks=10, markers=true, grid=false, c=3, legend=:bottomright)
    hline!(twinplt, [4, 8], lw=1, c=:black, ls=:dash, label="slope = [4, 8]")
    display(plt)
end

## Compare hamiltonian from M to the one from the diagonal_majoranas function at some time
sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6)
maj_hams1 = [1im * prod(diagonal_majoranas(prob, t))[5:8, 5:8] for t in prob[:ts]]
maj_hams2 = [1im * prod(diagonal_majoranas(prob, t))[1:4, 1:4] for t in prob[:ts]]
P1 = [I + 1im * prod(diagonal_majoranas(prob, t)[1:2])[5:8, 5:8] for t in prob[:ts]] / 2
P2 = [I + 1im * prod(diagonal_majoranas(prob, t)[1:2])[1:4, 1:4] for t in prob[:ts]] / 2
hams = [prob[:op](Matrix(I, 4, 4), prob[:p], t) for t in prob[:ts]]
projs = [eigen(Matrix(1im * ham)).vectors[:, 1:2] for ham in hams]
projs = [p * p' for p in projs]
pars = [sol(t)' * (1im * prod(diagonal_majoranas(prob, t)[1:2])[5:8, 5:8] * sol(t)) for t in prob[:ts]]

[abs(tr(P[:M, :L] * h)) for h in hams] |> plot
[abs(tr(P[:M, :L] * h)) for h in maj_hams1] |> plot!
[abs(tr(P[:M, :L] * h)) for h in maj_hams2] |> plot!
##
[abs(tr(p' * h1' * p * p' * h2 * p)) / (norm(p'h1 * p) * norm(p'h2 * p)) for (p, h1, h2) in zip(projs, hams, hams)] |> plot
[abs(tr(p' * h1' * p * p' * h2 * p)) / (norm(p'h1 * p) * norm(p'h2 * p)) for (p, h1, h2) in zip(projs, hams, maj_hams1)] |> plot!
[abs(tr(p' * h1' * p * p' * h2 * p)) / (norm(p'h1 * p) * norm(p'h2 * p)) for (p, h1, h2) in zip(projs, hams, maj_hams2)] |> plot!

##
[norm(p0 - p1) for (p0, p1) in zip(projs, P1)] |> plot
[norm(p0 - p1) for (p0, p1) in zip(projs, P2)] |> plot!
##
[abs(tr(P[0, 1] * h)) for h in hams] |> plot
[abs(tr(P[0, 2] * h)) for h in hams] |> plot!
[abs(tr(P[0, 3] * h)) for h in hams] |> plot!
[abs(tr(P[0, 3] * h)) for h in maj_hams1] |> plot!
[abs(tr(P[0, 2] * h)) for h in maj_hams1] |> plot!

## Do a sweep over zetas and plot the groundstate_components

zetas = range(1e-6, 1, length=100)
component_array = []
for (idx, ζ) in collect(enumerate(zetas))
    local_dict = Dict(
        :ζ => ζ,
        :ϵs => (0, 0, 0),
        :T => 1e4,
        :Δmax => 1 * [1 / 3, 1 / 2, 1],
        :Δmin => 1e-10 * [2, 1 / 3, 1],
        :k => 2e1,
        :steps => 4000,
        :correction => InterpolatedExactSimpleCorrection(),
        :interpolate_corrected_hamiltonian => false,
        :P => P,
        :inplace => inplace,
        :γ => γ,
        :u0 => U0
    )
    prob = setup_problem(local_dict)
    push!(component_array, MajoranaBraiding.analytic_parameters(find_zero_energy_from_analytics(ζ, prob[:ramp], prob[:ts][end] / 4, totalparity), ζ^2, prob[:ramp], 1 * prob[:ts][end] / 4))
end
## plot α/μ and ν/β as a log vs η
ηs = zetas .^ 2
plot(ηs, map(c -> c[:α] / c[:μ], component_array), label="α/μ", xlabel="η", ylabel="λ/η", lw=2, frame=:box)

function guess(η)
    b = -(√2 - 0.5) / (√2 - 1)
    println(b)
    e = b - 1 / 2
    println(e)
    return cos(atan(η))
    return (1 + b * η^2) / (1 + e * η^2)
end

function eta_von_x(x)
    u = 2 * (1 + x^2) / (2 + x^2)
    numerator = 2 - x^2 * u
    denominator = x^2 * u * (√(u) + x)^2
    return tan(acos(x))
end
x_array = range(1 / √2, 1, length=100)
η_array = [eta_von_x(x) for x in x_array]
plot!(η_array, x_array, label="analytics", lw=2, frame=:box)
