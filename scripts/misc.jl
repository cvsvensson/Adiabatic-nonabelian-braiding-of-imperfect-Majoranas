using MajoranaBraiding
using LinearAlgebra
using Plots
using ProgressMeter
using Base.Threads
##
param_dict = Dict(
    :η => 0.5,#(0.8, 0.4, 1), #Majorana overlaps. Number or triplet of numbers
    :T => 1e3, #Maximum time
    :k => 1e1, #Determines the slope of the ramp
    :steps => 500, #Number of timesteps for interpolations
    :correction => InterpolatedExactSimpleCorrection(), #Different corrections are available. This is the most relevant one for the paper
    # :correction => OptimizedSimpleCorrection(),
    # :correction => OptimizedIndependentSimpleCorrection(1, 0), 
    :interpolate_corrected_hamiltonian => true, #Creating an interpolated Hamiltonian might speed things up
    # :γ => γ, #Majorana basis
    :initial => (:L, :L̃) => 1, #Initial state. Use I for the identity matrix.
    :totalparity => -1, # The protocol works best for -1, as the gap closes for totalparity = 1 and η = 1
    :gapscaling => t -> 1 .+ 0.5 * cos(2pi * t) # Gap scaling function
)

## Solve the system
@time prob = setup_problem(param_dict);
visualize_protocol(prob)
##
@time sol = solve(prob);
plot(prob[:ts], stack([prob[:correction].scaling(t) for t in prob[:ts]]))
##
plot(sol.t, [(norm(sol(0.0)) - norm(sol(t))) for t in sol.t], label="norm error", xlabel="t") # get a sense of the numerical error
##
visualize_spectrum(prob)
visualize_rhos(prob)
visualize_deltas(prob)
visualize_analytic_parameters(prob)
visualize_protocol(prob)
visualize_parities(sol, prob)

## Heatmap of fidelity as a function of T and η
gridpoints = 40
T_arr = logrange(1e0, 5e3, length=gridpoints)
etas = range(1e-3, 1 - 1e-3, length=2 * gridpoints)
single_braid_fidelity = zeros(Float64, 2gridpoints, gridpoints)
double_braid_fidelity = zeros(Float64, 2gridpoints, gridpoints)
@time @showprogress for (idx_T, T) in enumerate(T_arr)
    Threads.@threads for (idx_z, η) in collect(enumerate(etas))
        local_dict = Dict(
            :η => η,
            :T => T,
            :k => 1e1,
            :steps => 200,
            :correction => InterpolatedExactSimpleCorrection(),
            :interpolate_corrected_hamiltonian => true,
            :totalparity => -1,
            :initial => I,
        )
        prob = setup_problem(local_dict)
        sol = solve(prob; saveat=[1, 2])
        proj = prob[:totalparity] == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
        # proj = Diagonal([1, 0, 0, 1])
        single_braid_gate = majorana_exchange(-prob[:P][:L, :R])
        # single_braid_gate = analytical_protocol_gate(prob)
        double_braid_gate = single_braid_gate^2
        single_braid_result = sol(1)
        double_braid_result = sol(2)
        single_braid_fidelity[idx_z, idx_T] = gate_fidelity(proj * single_braid_gate * proj, proj * single_braid_result * proj)
        double_braid_fidelity[idx_z, idx_T] = gate_fidelity(proj * double_braid_gate * proj, proj * double_braid_result * proj)
    end
end
plot(heatmap(T_arr, etas, single_braid_fidelity .^ 2, xlabel="T", ylabel="η", c=:viridis, title="Single braid fidelity", clim=(0, 1), xscale=:log),
    heatmap(T_arr, etas, double_braid_fidelity .^ 2, xlabel="T", ylabel="η", c=:viridis, title="Double braid fidelity", clim=(0, 1)), xscale=:log)


## 1d sweep over eta for the fidelity
gridpoints = 40
ηs = (range(0, 1, gridpoints)) #range(0, 1, length=gridpoints)
double_braid_majorana_fidelity = zeros(Float64, gridpoints)
double_braid_kato_fidelity = zeros(Float64, gridpoints)
double_braid_analytical_gate_fidelity = zeros(Float64, gridpoints)
analytical_fidelity = zeros(Float64, gridpoints)
fidelities = zeros(Float64, gridpoints)
fidelity_numerics_analytic = zeros(Float64, gridpoints)
@time @showprogress @threads for (idx, η) in collect(enumerate(ηs))
    local_dict = Dict(
        :η => η .* (1, 1, 1),
        :T => 1e4,
        :k => 1e1,
        :steps => 1000,
        :correction => InterpolatedExactSimpleCorrection(),
        # :correction => OptimizedSimpleCorrection(),
        # :correction => OptimizedIndependentSimpleCorrection(30, 1e-2),
        # :correction => NoCorrection(),
        # :correction => SimpleCorrection(),
        :interpolate_corrected_hamiltonian => true,
        :initial => I,
        :totalparity => -1
    )
    prob = setup_problem(local_dict)
    sol = solve(prob; saveat=[2])
    proj = prob[:totalparity] == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
    majorana_double_braid = majorana_exchange(prob[:P][:L, :R])^2
    double_kato = single_braid_gate_kato(prob)^2
    double_gate_analytical = single_braid_gate_analytical(prob)^2
    double_braid_result = sol(2)
    double_braid_majorana_fidelity[idx] = gate_fidelity(majorana_double_braid, double_braid_result, proj)
    double_braid_kato_fidelity[idx] = gate_fidelity(double_kato, double_braid_result, proj)
    double_braid_analytical_gate_fidelity[idx] = gate_fidelity(double_gate_analytical, majorana_double_braid, proj)
    fidelity_numerics_analytic[idx] = gate_fidelity(double_gate_analytical, double_braid_result, proj)
    analytical_fidelity[idx] = analytical_gate_fidelity(prob)
end
##
plot(ηs, double_braid_majorana_fidelity, lw=2, marker=true, ylims=(-0.01, 1.01), label="fidelity to majorana gate")
plot!(ηs, analytical_fidelity, xlabel="η", ylabel="Fidelity", lw=2, frame=:box, label="analytical solution in the article")
plot!(ηs, fidelity_numerics_analytic, lw=2, label="fidelity to analytical gate", marker=true)
##
plot(; xlabel="η", lw=2, frame=:box)
plot(ηs, double_braid_majorana_fidelity, label="double_braid_majorana_fidelity", lw=2)
plot!(ηs, double_braid_kato_fidelity, label="double_braid_kato_fidelity", lw=2)
plot!(ηs, analytical_fidelity, label="analytical majorana similarity", lw=2)
plot!(ηs, double_braid_analytical_gate_fidelity, label="double_braid_analytical_gate_fidelity", lw=2)


## Check that the analytical gapped majoranas project on the ground state
param_dict = Dict(
    :η => 0.2, #Majorana overlaps. Number or triplet of numbers
    :T => 5e3, #Maximum time
    :k => 2e1, #Determines the slope of the ramp
    :steps => 2000, #Number of timesteps for interpolations
    :correction => InterpolatedExactSimpleCorrection(),
    :interpolate_corrected_hamiltonian => true,
    :γ => γ, #Majorana basis
    :initial => I,
    :totalparity => -1
)
prob = setup_problem(param_dict)
subinds = totalparity == 1 ? (5:8) : (1:4)
hams = [prob[:H](prob[:p], t) for t in prob[:ts]]
projs = [eigen(Matrix(ham)).vectors[:, 1:2] for ham in hams]
projs = [p * p' for p in projs]
P = [I - 1im * prod(diagonal_majoranas(prob, t)[1:2])[subinds, subinds] for t in prob[:ts]] / 2
[norm(p0 - p1) for (p0, p1) in zip(projs, P)] |> norm < 1e-12
