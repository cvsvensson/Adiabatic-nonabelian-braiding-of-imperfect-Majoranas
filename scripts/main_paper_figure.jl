using MajoranaBraiding
using LinearAlgebra
using Plots
using ProgressMeter
using Base.Threads
using LaTeXStrings
## Plot settings
default(fontfamily="Computer Modern", linewidth=2, framestyle=:box, label=nothing, grid=false)
colors = ["#FFC000", "#00B0F0", "#92D050"]
##
gridpoints = 2000
ηs = (range(0, 1, gridpoints))
double_braid_majorana_fidelity = zeros(Float64, gridpoints)
analytical_fidelity = zeros(Float64, gridpoints)
uncorrected_double_braid_majorana_fidelity = zeros(Float64, gridpoints)
@time @showprogress @threads :static for (n, η) in collect(enumerate(ηs))
    local_dict = Dict(
        :η => η,
        :T => 2e2,
        :k => 1e1,
        :steps => 1000,
        :correction => InterpolatedExactSimpleCorrection(),
        # :correction => OptimizedSimpleCorrection(),
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

    local_dict[:correction] = NoCorrection()
    uncorrected_sol = solve(setup_problem(local_dict); saveat=[2])
    uncorrected_double_braid_majorana_fidelity[n] = gate_fidelity(majorana_double_braid, uncorrected_sol(2), proj)
end

## Figure for the paper 
# ylabel=L"MBS Similarity $S$"
# ylabel = L"S"
# tickfontsize = 9
# legendfontsize = 9
# ylabelfontsize = 10
# xlabelfontsize = 15
xticks = xticks = ([0, 1 / 2, 1], ["0", "0.5", "1"])
yticks = ([0, 1 / 2, 1], ["0", L"\frac{1}{2}", "1"])
# p = plot(; frame=:box, size=0.55 .* (600, 250), xlabelfontsize, ylabelfontsize, legendfontsize, ylims=(-0.03, 1.03), yticks, xticks, legendposition=:topright, margin=0Plots.mm, right_margin=1Plots.mm, tickfontsize=9, xticksize=0)
p = plot(; frame=:box, size=0.55 .* (600, 250), ylims=(-0.03, 1.03), yticks, xticks, legendposition=:topright, margin=0Plots.mm, right_margin=1Plots.mm, thickness_scaling=1.1)
plot!(p, ηs, uncorrected_double_braid_majorana_fidelity, label="Uncorrected", lw=2, c=colors[2])
plot!(p, ηs, analytical_fidelity, lw=3, label="Corrected", c=colors[3])
# annotate!(p, 0.6, -0.1, L"$\eta$")
annotate!(p, 1.055, -0.09, L"$\eta$", 12)
annotate!(p, -0.1, 1, L"S", 12)
##
savefig(p, "majorana_similarity.pdf")
