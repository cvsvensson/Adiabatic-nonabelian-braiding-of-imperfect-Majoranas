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
use_static_arrays = true
mtype, vtype = SMatrix{2^(N - 1),2^(N - 1)}, SVector{2^(N - 1)}

## Initial state and identity matrix
# u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[:M, :M̃] + P[:L, :L̃] + P[:R, :R̃]), 1:1).vectors))))
U0 = mtype(Matrix{ComplexF64}(I, 2^(N - 1), 2^(N - 1)))

##
param_dict = Dict(
    :ζ => 0.2, #Majorana overlaps. Number or triplet of numbers
    :ϵs => (0, 0, 0), #Dot energy levels
    :T => 1e4, #Maximum time
    :Δmax => 1 * (rand(3) .+ 0.5), #Largest values of Δs. Number or triplet of numbers
    :Δmin => 1e-10 * (rand(3) .+ 0.5), #Smallest values of Δs. Number or triplet of numbers
    :k => 1e1, #Determines the slope of the ramp
    :steps => 2000, #Number of timesteps for interpolations
    :correction => InterpolatedExactSimpleCorrection(), #Different corrections are available. This is the most relevant one for the paper
    :interpolate_corrected_hamiltonian => true, #Creating an interpolated Hamiltonian might speed things up
    :γ => γ, #Majorana basis
    :u0 => U0, #Initial state. Use U0 for the identity matrix.
    :extra_shifts => [0, 0, 0], #Shifts the three Δ pulses. Given as fractions of T
    :totalparity => -1
)

## Solve the system
prob = setup_problem(param_dict)
@time sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6);
plot(sol.t, [(norm(sol(0.0)) - norm(sol(t))) for t in sol.t], label="norm error", xlabel="t")
##
visualize_spectrum(prob)
visualize_deltas(prob)
visualize_parities(sol, prob)
visualize_analytic_parameters(prob)
visualize_protocol(prob)

## Calculate full solution for T and 2T and calculate the fidelities
gridpoints = 10
T_arr = range(1e1, 1e3, length=gridpoints)
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
            :interpolate_corrected_hamiltonian => true,
            :γ => γ,
            :totalparity => -1,
            :u0 => U0
        )
        prob = setup_problem(local_dict)
        sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6, saveat=[0, T, 2T])
        proj = prob[:totalparity] == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
        # proj = Diagonal([1, 0, 0, 1])
        single_braid_gate = majorana_exchange(-prob[:P][:L, :R])
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
        :T => 5e3,
        :Δmax => 1 * [1 / 3, 1 / 2, 1],
        :Δmin => 1e-10 * [2, 1 / 3, 1],
        :k => 1e1,
        :steps => 4000,
        :correction => InterpolatedExactSimpleCorrection(),
        # :correction => EigenEnergyCorrection(),
        # :correction => NoCorrection(),
        # :correction => SimpleCorrection(),
        :interpolate_corrected_hamiltonian => true,
        :γ => γ,
        :u0 => U0,
        :totalparity => -1
    )
    T = local_dict[:T]
    prob = setup_problem(local_dict)
    sol = solve(prob[:odeprob], Tsit5(), abstol=1e-8, reltol=1e-8, saveat=[0, T, 2T])
    proj = prob[:totalparity] == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
    single_braid_gate_ideal = majorana_exchange(-prob[:P][:L, :R])
    double_braid_gate_ideal = single_braid_gate_ideal^2
    single_braid_result = sol(T)
    double_braid_result = sol(2T)
    analytical_angles[idx] = single_braid_gate_analytical_angle(prob)
    single_braid_ideal_fidelity[idx] = gate_fidelity(proj * single_braid_gate_ideal * proj, proj * single_braid_result * proj)
    double_braid_ideal_fidelity[idx] = gate_fidelity(proj * double_braid_gate_ideal * proj, proj * double_braid_result * proj)
    analytical_fidelity[idx] = analytical_gate_fidelity(prob)
end
##
plot(; xlabel="ω", lw=2, frame=:box, ylims=(-0.02, 1.02))
plot!(omegas, single_braid_ideal_fidelity, label="single_braid_ideal_fidelity", xlabel="ω", lw=2, markers=true)
plot!(omegas, double_braid_ideal_fidelity, label="double_braid_ideal_fidelity", lw=2)
plot!(omegas, analytical_fidelity, label="analytical majorana similarity", lw=2)
## plot angles 
plot(omegas, analytical_angles, label="analytical_angles", lw=2, frame=:box)
##
## Compare hamiltonian from M to the one from the diagonal_majoranas function at some time
sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6)
maj_hams1 = [1im * prod(diagonal_majoranas(prob, t))[5:8, 5:8] for t in prob[:ts]]
maj_hams2 = [1im * prod(diagonal_majoranas(prob, t))[1:4, 1:4] for t in prob[:ts]]
P1 = [I - 1im * prod(diagonal_majoranas(prob, t)[1:2])[5:8, 5:8] for t in prob[:ts]] / 2
P2 = [I + 1im * prod(diagonal_majoranas(prob, t)[1:2])[1:4, 1:4] for t in prob[:ts]] / 2
hams = [prob[:op](Matrix(I, 4, 4), prob[:p], t) for t in prob[:ts]]
projs = [eigen(Matrix(1im * ham)).vectors[:, 1:2] for ham in hams]
projs = [p * p' for p in projs]

if prob[:totalparity] == 1
    pars = [sol(t)' * (1im * prod(diagonal_majoranas(prob, t)[1:2])[5:8, 5:8] * sol(t)) for t in prob[:ts]]
    P = [I + 1im * prod(diagonal_majoranas(prob, t)[1:2])[5:8, 5:8] for t in prob[:ts]] / 2
else
    pars = [sol(t)' * (1im * prod(diagonal_majoranas(prob, t)[1:2])[1:4, 1:4] * sol(t)) for t in prob[:ts]]
    P = [I + 1im * prod(diagonal_majoranas(prob, t)[1:2])[1:4, 1:4] for t in prob[:ts]] / 2
end
# Plot pars
real(pars) |> plot
##
[norm(p0 - p1) for (p0, p1) in zip(projs, P)] |> plot
##
[abs(tr(prob[:P][:M, :L] * h)) for h in hams] |> plot
[abs(tr(prob[:P][:M, :L] * h)) for h in maj_hams1] |> plot!
[abs(tr(prob[:P][:M, :L] * h)) for h in maj_hams2] |> plot!
##
[abs(tr(p' * h1' * p * p' * h2 * p)) / (norm(p'h1 * p) * norm(p'h2 * p)) for (p, h1, h2) in zip(projs, hams, hams)] |> plot
[abs(tr(p' * h1' * p * p' * h2 * p)) / (norm(p'h1 * p) * norm(p'h2 * p)) for (p, h1, h2) in zip(projs, hams, maj_hams1)] |> plot!
[abs(tr(p' * h1' * p * p' * h2 * p)) / (norm(p'h1 * p) * norm(p'h2 * p)) for (p, h1, h2) in zip(projs, hams, maj_hams2)] |> plot!

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
        :γ => γ,
        :u0 => U0,
        :totalparity => -1
    )
    prob = setup_problem(local_dict)
    push!(component_array, MajoranaBraiding.analytic_parameters(find_zero_energy_from_analytics(ζ, prob[:ramp], prob[:ts][end] / 4, 0.0, prob[:totalparity]), ζ^2, prob[:ramp], 1 * prob[:ts][end] / 4))
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
