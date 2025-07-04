using MajoranaBraiding
using LinearAlgebra
using Plots
using ProgressMeter
using Base.Threads
using LaTeXStrings
## Plot settings
# gr() # Faster
pgfplotsx() # Needs LaTeX installation
colors = ["#FFC000", "#00B0F0", "#92D050"]
# colors = 1:3 # More contrast
##
gridpoints = 2000
ηs = (range(0, 1, gridpoints))
double_braid_majorana_fidelity = zeros(Float64, gridpoints)
analytical_fidelity = zeros(Float64, gridpoints)
uncorrected_double_braid_majorana_fidelity = zeros(Float64, gridpoints)
@time @showprogress @threads for (n, η) in collect(enumerate(ηs))
    local_dict = Dict(
        :η => η,
        :T => 2e2,
        :k => 1e1,
        :steps => 1000,
        :correction => InterpolatedExactSimpleCorrection(),
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
## Check that the corrected protocol agrees with the analytical fidelity
p = plot(; frame=:box, ylabel=L"MBS Similarity $S$", xlabel=L"\eta", size=0.7 .* (600, 400), xlabelfontsize=15, legendfontsize=9, ylims=(-0.03, 1.03), yticks=([0, 1 / 2, 1], ["0", L"\frac{1}{2}", "1"]), xticks=([0, 1 / 2, 1], ["0", "0.5", "1"]), legendposition=:topright)
plot!(p, ηs, analytical_fidelity, lw=3, label="Corrected: adiabatic", c=colors[3])
plot!(p, ηs, double_braid_majorana_fidelity, lw=3, label="Corrected: finite time", ls=:dash, c=colors[1])
plot!(p, ηs, uncorrected_double_braid_majorana_fidelity, label="Uncorrected", seriestype=:stepmid, lw=1, c=colors[2])
##
savefig(p, "majorana_similarity_SI.pdf")

## Figure for the paper 
p = plot(; frame=:box, ylabel=L"MBS Similarity $S$", xlabel=L"\eta", size=0.6 .* (600, 400), xlabelfontsize=15, legendfontsize=9, ylims=(-0.03, 1.03), yticks=([0, 1 / 2, 1], ["0", L"\frac{1}{2}", "1"]), xticks=([0, 1 / 2, 1], ["0", "0.5", "1"]), legendposition=:topright, tex_output_standalone=true)
plot!(p, ηs, uncorrected_double_braid_majorana_fidelity, label="Uncorrected", seriestype=:stepmid, lw=1, c=colors[2])
plot!(p, ηs, analytical_fidelity, lw=2, label="Corrected", c=colors[3])
##
savefig(p, "majorana_similarity.pdf")
# plot!(p, ηs, double_braid_majorana_fidelity, lw=3, label="Corrected", marker=false, ls=:dash, c=colors[3])