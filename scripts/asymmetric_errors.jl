using MajoranaBraiding
using LinearAlgebra
using Plots
using ProgressMeter
using Base.Threads
using LaTeXStrings
## Plot settings
# gr() # Faster
pgfplotsx() # Needs LaTeX installation
colors = ["#FFC000", "#00B0F0", "#92D050"] # Matches existing figures
#colors = 1:3 # More contrast

## Run simulation
gridpoints = 20
ηs = (range(0, 1.5, gridpoints)) #range(0, 1, length=gridpoints)
double_braid_majorana_fidelity = zeros(Float64, gridpoints)
analytical_fidelity = zeros(Float64, gridpoints)
asymmetry = (1, 0.5, 0.2)
effective_η_scaling = MajoranaBraiding.effective_η(asymmetry)
@time @showprogress @threads for (n, η) in collect(enumerate(ηs))
    local_dict = Dict(
        :η => η .* asymmetry,
        :T => 2e3,
        :k => 1e1,
        :steps => 1000,
        # :correction => InterpolatedExactSimpleCorrection(),
        # :correction => OptimizedSimpleCorrection(),
        :correction => OptimizedIndependentSimpleCorrection(100, 1e-2),
        :interpolate_corrected_hamiltonian => true,
        :initial => I,
        :totalparity => -1
    )
    prob = setup_problem(local_dict)
    sol = solve(prob; saveat=[2])
    proj = prob[:totalparity] == 1 ? Diagonal([0, 1, 1, 0]) : Diagonal([1, 0, 0, 1])
    majorana_double_braid = majorana_exchange(prob[:P][:L, :R])^2
    double_braid_majorana_fidelity[n] = gate_fidelity(majorana_double_braid, sol(2), proj)
    analytical_fidelity[n] = analytical_gate_fidelity(prob)
end

## Plot results
effective_ηs = effective_η_scaling * ηs
p = plot(; frame=:box, ylabel=L"MBS Similarity $S$", xlabel=L"\eta_\mathrm{eff} = \sqrt{\eta_1\sqrt{\eta_2\eta_3}}", size=0.7 .* (600, 400), xlabelfontsize=15, legendfontsize=10, ylims=(-0.03, 1.03), legendposition=:bottomleft)
plot!(p, effective_ηs, analytical_fidelity, lw=3, label=L"Analytic curve with $\eta_\mathrm{eff}$", c=colors[3])
scatter!(p, effective_ηs, double_braid_majorana_fidelity, label="Asymmetric correction protocol", c=colors[1], marker=true)
##
savefig(p, "majorana_similarity_asymmetric.pdf")
# savefig(p, "majorana_similarity_asymmetric.tex")
# savefig(p, "majorana_similarity_asymmetric.tikz")
