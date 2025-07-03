using MajoranaBraiding
using LinearAlgebra
using Plots
using ProgressMeter
using Base.Threads
using LaTeXStrings
##
gridpoints = 30
ηs = (range(0, 1.1, gridpoints)) #range(0, 1, length=gridpoints)
double_braid_majorana_fidelity = zeros(Float64, gridpoints)
double_braid_kato_fidelity = zeros(Float64, gridpoints)
double_braid_analytical_gate_fidelity = zeros(Float64, gridpoints)
angles = zeros(Float64, gridpoints)
analytical_fidelity = zeros(Float64, gridpoints)
fidelities = zeros(Float64, gridpoints)
fidelity_numerics_analytic = zeros(Float64, gridpoints)
uncorrected_double_braid_majorana_fidelity = zeros(Float64, gridpoints)
asymmetry = (1.4, 0.7, 0.3)
effective_η_scaling = MajoranaBraiding.effective_η(asymmetry)
@time @showprogress @threads for (n, η) in collect(enumerate(ηs))
    local_dict = Dict(
        :η => η .* asymmetry,#η .* (1.2, 0.8, 0.6),
        :T => 1e3,
        :Δmax => 1 * [1 / 3, 1 / 2, 1],
        :Δmin => 0 * 1e-10 * [2, 1 / 3, 1],
        :k => 1e1,
        :steps => 1000,
        # :correction => InterpolatedExactSimpleCorrection(),
        # :correction => OptimizedSimpleCorrection(),
        :correction => OptimizedIndependentSimpleCorrection(100, 1e-2),
        :interpolate_corrected_hamiltonian => true,
        :initial => I,
        :totalparity => -1
    )
    T = local_dict[:T]
    prob = setup_problem(local_dict)
    sol = solve(prob; saveat=[2T])
    proj = prob[:totalparity] == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
    majorana_double_braid = majorana_exchange(prob[:P][:L, :R])^2
    double_kato = single_braid_gate_kato(prob)^2
    double_gate_analytical = single_braid_gate_analytical(prob)^2
    # double_braid_result = sol(2T)
    double_braid_majorana_fidelity[n] = gate_fidelity(majorana_double_braid, sol(2T), proj)
    double_braid_kato_fidelity[n] = gate_fidelity(double_kato, sol(2T), proj)
    double_braid_analytical_gate_fidelity[n] = gate_fidelity(double_gate_analytical, majorana_double_braid, proj)
    fidelity_numerics_analytic[n] = gate_fidelity(double_gate_analytical, sol(2T), proj)
    analytical_fidelity[n] = analytical_gate_fidelity(prob)

    local_dict[:correction] = NoCorrection()
    uncorrected_sol = solve(setup_problem(local_dict); saveat=[2T])
    uncorrected_double_braid_majorana_fidelity[n] = gate_fidelity(majorana_double_braid, uncorrected_sol(2T), proj)
end
##
default(; :fontfamily => "Computer Modern")
p = plot(; frame=:box, ylabel=L"S", xlabel=L"\tilde{\eta} = \sqrt{η_1\sqrt{\eta_2\eta_3}}", size=0.8 .* (600, 400), labelfontsize=15, title="Asymmetric correction protocol", ylims=(-0.03, 1.03))
effective_ηs = effective_η_scaling * ηs
# plot!(p, effective_ηs, uncorrected_double_braid_majorana_fidelity, label="Uncorrected"; marker=false, lw=2)
plot!(p, effective_ηs, double_braid_majorana_fidelity, lw=3, label="Corrected", marker=true, ls=:dash)
plot!(p, effective_ηs, analytical_fidelity, lw=3, label=L"S(\tilde{η})", marker=false)
# plot!(p, effective_ηs, effective_ηs, lw=3, label="effective η", marker=false)
# plot!(p, effective_ηs, ηs, lw=3, label="η", marker=false)