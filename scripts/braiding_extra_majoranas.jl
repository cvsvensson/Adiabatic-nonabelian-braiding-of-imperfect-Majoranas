using MajoranaBraiding
using QuantumDots
using Majoranas
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using ProgressMeter
using StaticArrays
using Base.Threads
# using Roots
using Accessors

##
nbr_of_majoranas = 6
N = nbr_of_majoranas ÷ 2
majorana_labels = 0:5
γ = SingleParticleMajoranaBasis(nbr_of_majoranas, majorana_labels)
totalparity = 1
use_static_arrays = true
inplace = !use_static_arrays
mtype, vtype = MajoranaBraiding.matrix_vec_types(use_static_arrays, inplace, N)
P = parity_operators(γ, totalparity, mtype)
# H = ham_with_corrections
## Initial state and identity matrix
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[0, 1] + P[2, 4] + P[3, 5]), 1:1).vectors))))
U0 = mtype(Matrix{ComplexF64}(I, size(u0, 1), size(u0, 1)))

##
param_dict = Dict(
    :ζ => 0.7, #Majorana overlaps. Number or triplet of numbers
    :ϵs => (0, 0, 0), #Dot energy levels
    :T => 2e3, #Maximum time
    :Δmax => 1 * (rand(3) .+ 0.5), #Largest values of Δs. Number or triplet of numbers
    :Δmin => 1e-6 * (rand(3) .+ 0.5), #Smallest values of Δs. Number or triplet of numbers
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

function setup_problem(dict)
    extra_shifts = get(dict, :extra_shifts, @SVector [0, 0, 0])
    ramp = RampProtocol(dict[:Δmin], dict[:Δmax], dict[:T], dict[:k], extra_shifts)
    tspan = (0.0, 2 * dict[:T])
    ts = range(0, tspan[2], dict[:steps])
    newdict = Dict(dict..., :ramp => ramp, :ts => ts, :tspan => tspan)
    corr = MajoranaBraiding.setup_correction(dict[:correction], newdict)
    p = (ramp, dict[:ϵs], dict[:ζ], corr, dict[:P])
    interpolate = get(dict, :interpolate_corrected_hamiltonian, false)
    M = interpolate ? MajoranaBraiding.get_iH_interpolation_op(ham_with_corrections, p, ts) : get_op(ham_with_corrections, p)
    prob = ODEProblem{dict[:inplace]}(M, dict[:u0], tspan, p)
    newdict = Dict(newdict..., :correction => corr, :p => p)
    return (; odeprob=prob, dict=newdict, op=M, p, ts, T=dict[:T])
end
## Solve the system
prob = setup_problem(param_dict)
@time sol = solve(prob.odeprob, Tsit5(), abstol=1e-6, reltol=1e-6);
plot(sol.t, [1 .- norm(sol(t)) for t in sol.t], label="norm error", xlabel="t")
##
visualize_spectrum(prob.dict)
visualize_deltas(prob.dict)
visualize_parities(sol, prob.dict)
visualize_groundstate_components(prob.dict)
visualize_protocol(prob.dict)
##
full_gate_param_dict = @set param_dict[:u0] = U0
prob_full = setup_problem(full_gate_param_dict)#ODEProblem{inplace}(M, U0, tspan, p)
@time sol_full = solve(prob_full.odeprob, Tsit5(), reltol=1e-6, abstol=1e-6);
single_braid_gate = majorana_exchange(-P[2, 3])
single_braid_gate = single_braid_gate_kato(prob_full.dict)
double_braid_gate = single_braid_gate^2
single_braid_result = sol_full(prob_full.dict[:T])
double_braid_result = sol_full(2prob_full.dict[:T])
proj = Diagonal([0, 1, 1, 0])
single_braid_fidelity = gate_fidelity(proj * single_braid_gate * proj, proj * single_braid_result * proj)
println("Single braid fidelity: ", single_braid_fidelity)
double_braid_fidelity = gate_fidelity(proj * double_braid_gate * proj, proj * double_braid_result * proj)
println("Double braid fidelity: ", double_braid_fidelity)

println("Fit of angle for braid gate: ", braid_gate_best_angle(single_braid_gate, P))

braid_gate_prediction(single_braid_gate, single_braid_gate_analytical_angle(prob_full.dict), P)
##
# Do a sweep over several zetas, solve the system for the final time t=2T and measure the parities
zetas = range(0, 1, length=50)
parity_measurements = [(0, 1), (2, 4), (3, 5)]
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
        :inplace => inplace,
        :γ => γ,
        :u0 => u0
    )
    prob = setup_problem(local_dict)
    sol = solve(prob.odeprob, Tsit5(), abstol=1e-6, reltol=1e-6)
    parities_arr[idx, :] = measure_parities(sol(2prob.T), prob.dict, parity_measurements)
end
plot(zetas, real(parities_arr), label=permutedims(parity_measurements), xlabel="ζ", ylabel="Parity", lw=2, frame=:box)
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
        sol = solve(prob.odeprob, Tsit5(), abstol=1e-6, reltol=1e-6)
        parities_after_T_2D[idx_z, idx_T, :] = measure_parities(sol(prob.T), prob.dict, parity_measurements)
        parities_arr_2D[idx_z, idx_T, :] = measure_parities(sol(2prob.T), prob.dict, parity_measurements)
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
        sol = solve(prob.odeprob, Tsit5(), abstol=1e-6, reltol=1e-6, saveat=[0, T, 2T])
        proj = Diagonal([0, 1, 1, 0])
        # proj = Diagonal([1, 0, 0, 1])
        single_braid_gate = majorana_exchange(-P[2, 3])
        # single_braid_gate = single_braid_gate_kato(prob.dict)
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
gridpoints = 40
zetas = range(0.001, 1, length=gridpoints)
single_braid_ideal_fidelity = zeros(Float64, gridpoints)
single_braid_kato_fidelity = zeros(Float64, gridpoints)
double_braid_ideal_fidelity = zeros(Float64, gridpoints)
double_braid_kato_fidelity = zeros(Float64, gridpoints)
angles = zeros(Float64, gridpoints)
analytical_angles = zeros(Float64, gridpoints)
fidelities = zeros(Float64, gridpoints)
fidelity_numerics_analytic = zeros(Float64, gridpoints)
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
    sol = solve(prob.odeprob, Tsit5(), abstol=1e-8, reltol=1e-8, saveat=[0, T, 2T])
    proj = totalparity == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
    single_braid_gate_ideal = majorana_exchange(-P[2, 3])
    single_braid_gate_kato_ = single_braid_gate_kato(prob.dict)
    double_braid_gate_ideal = single_braid_gate_ideal^2
    double_braid_gate_kato = single_braid_gate_kato_^2
    single_braid_result = sol(T)
    double_braid_result = sol(2T)
    analytical_angles[idx] = single_braid_gate_analytical_angle(prob.dict)
    angles[idx] = braid_gate_best_angle(single_braid_result, P)[1]
    fidelities[idx] = braid_gate_best_angle(single_braid_result, P)[2]
    single_braid_ideal_fidelity[idx] = gate_fidelity(proj * single_braid_gate_ideal * proj, proj * single_braid_result * proj)
    single_braid_kato_fidelity[idx] = gate_fidelity(proj * single_braid_gate_kato_ * proj, proj * single_braid_result * proj)
    double_braid_ideal_fidelity[idx] = gate_fidelity(proj * double_braid_gate_ideal * proj, proj * double_braid_result * proj)
    double_braid_kato_fidelity[idx] = gate_fidelity(proj * double_braid_gate_kato * proj, proj * double_braid_result * proj)

    fidelity_numerics_analytic[idx] = gate_fidelity(proj * single_braid_gate * proj, proj * MajoranaBraiding.single_braid_gate_fit(angles[idx], P) * proj)
end
##
#plot(zetas, single_braid_ideal_fidelity, label="single_braid_ideal_fidelity", xlabel="ζ", lw=2, frame=:box)
#plot!(zetas, double_braid_ideal_fidelity, label="double_braid_ideal_fidelity", lw=2, frame=:box)
plot(zetas, single_braid_kato_fidelity, label="single_braid_kato_fidelity", lw=2, frame=:box)
plot!(zetas, double_braid_kato_fidelity, label="double_braid_kato_fidelity", lw=2, frame=:box)
plot!(zetas, 1 .- (angles .- analytical_angles) .^ 2, label="1- (angles - analytical_angles)^2", xlabel="ζ", lw=2, frame=:box)
## plot angles 
plot(zetas, angles, label="angles", xlabel="ζ", lw=2, frame=:box)
plot!(zetas, analytical_angles, label="analytical_angles", lw=2, frame=:box)
##
plot(zetas, single_braid_kato_fidelity, label="single_braid_kato_fidelity", lw=2, frame=:box)
plot!(zetas, double_braid_kato_fidelity, label="double_braid_kato_fidelity", lw=2, frame=:box)


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

maj_hams1 = [1im * prod(diagonal_majoranas(prob.dict, t))[5:8, 5:8] for t in prob.ts]
maj_hams2 = [1im * prod(diagonal_majoranas(prob.dict, t))[1:4, 1:4] for t in prob.ts]
hams = [prob.op(Matrix(I, 4, 4), prob.p, t) for t in prob.ts]

[abs(tr(P[0, 2] * h)) for h in hams] |> plot
[abs(tr(P[0, 2] * h)) for h in maj_hams1] |> plot!
[abs(tr(P[0, 2] * h)) for h in maj_hams2] |> plot!