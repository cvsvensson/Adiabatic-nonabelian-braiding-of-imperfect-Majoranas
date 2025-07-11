using MajoranaBraiding
using LinearAlgebra
using Plots
using ProgressMeter
using Base.Threads
using LaTeXStrings
using Statistics
import Random: seed!
seed!(1)
## Plot settings
default(fontfamily="Computer Modern", linewidth=2, framestyle=:box, label=nothing, grid=false)
colors = ["#FFC000", "#00B0F0", "#92D050"]

## Statistics over different final times
gridpoints = 400
realizations = 400
ηs = range(0, 1, gridpoints)
Ts = [T0 .* (1 .+ (rand(realizations) .- 0.5) .* 0.2) for T0 in [50, 200]]
double_braid_majorana_fidelity_statistics = zeros(Float64, gridpoints, realizations, 2)
analytical_fidelity_statistics = zeros(Float64, gridpoints, realizations, 2)
uncorrected_double_braid_majorana_fidelity_statistics = zeros(Float64, gridpoints, realizations, 2)
@time @showprogress for (n, η) in enumerate(ηs)
    for (k, _Ts) in enumerate(Ts)
        @threads :static for (m, T) in collect(enumerate(_Ts))
            local_dict = Dict(
                :η => η,
                :T => T,
                :k => 10,
                :steps => 500,
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
            double_braid_majorana_fidelity_statistics[n, m, k] = gate_fidelity(majorana_double_braid, sol(2), proj)
            analytical_fidelity_statistics[n, m, k] = analytical_gate_fidelity(prob)

            local_dict[:correction] = NoCorrection()
            uncorrected_sol = solve(setup_problem(local_dict); saveat=[2])
            uncorrected_double_braid_majorana_fidelity_statistics[n, m, k] = gate_fidelity(majorana_double_braid, uncorrected_sol(2), proj)
        end
    end
end

## Plots
ribbonfunc = (m; dims) -> begin # this function uses std for the ribbon, but clamps it so that it does not go below 0
    me = dropdims(mapslices(mean, m; dims); dims)
    s = dropdims(mapslices(std, m; dims); dims)
    (map((me, s) -> me - s < 0 ? abs(me) : s, me, s), s)
end
yticks = ([0, 1 / 2, 1], ["0", L"\frac{1}{2}", "1"])
ylims = (-0.03, 1.03)
xticks = ([0, 1 / 2, 1], ["0", "0.5", "1"])

# Short time plot
p_shorter_time = plot(; ylims, yticks, xticks=false, legend=false)
plot!(p_shorter_time, ηs, analytical_fidelity_statistics[:, 1, 1], lw=3, label="Corrected: adiabatic", c=colors[3])
plot!(p_shorter_time, ηs, mean(double_braid_majorana_fidelity_statistics[:, :, 1], dims=2), lw=3, label="Corrected: finite time", ls=:dash, c=colors[1], ribbon=ribbonfunc(double_braid_majorana_fidelity_statistics[:, :, 1], dims=2))
plot!(p_shorter_time, ηs, mean(uncorrected_double_braid_majorana_fidelity_statistics[:, :, 1], dims=2), label="Uncorrected", lw=2, c=colors[2], ribbon=ribbonfunc(uncorrected_double_braid_majorana_fidelity_statistics[:, :, 1], dims=2))
annotate!(p_shorter_time, -0.11, 1, text(L"\mathrm{(a)}", 10))
annotate!(p_shorter_time, -0.11, 0.8, text(L"S", 10))

# Longer time plot
p_long_time = plot(; xlabel=L"\eta", ylims, yticks, xticks, legendposition=:topright)
plot!(p_long_time, ηs, analytical_fidelity_statistics[:, 1, 1], lw=3, label="Corrected: adiabatic", c=colors[3])
plot!(p_long_time, ηs, mean(double_braid_majorana_fidelity_statistics[:, :, 2], dims=2), lw=3, label="Corrected: finite time", ls=:dash, c=colors[1], ribbon=ribbonfunc(double_braid_majorana_fidelity_statistics[:, :, 2], dims=2))
plot!(p_long_time, ηs, mean(uncorrected_double_braid_majorana_fidelity_statistics[:, :, 2], dims=2), label="Uncorrected", lw=2, c=colors[2], ribbon=ribbonfunc(uncorrected_double_braid_majorana_fidelity_statistics[:, :, 2], dims=2))
annotate!(p_long_time, -0.11, 1, text(L"\mathrm{(b)}", 10))
annotate!(p_long_time, -0.11, 0.8, text(L"S", 10))
#
p_statistics = plot(p_shorter_time, p_long_time, layout=(2, 1), size=0.6 .* (600, 500), thickness_scaling=1.2, margin=0Plots.mm, bottom_margin=[-3Plots.mm -3Plots.mm], left_margin=1Plots.mm, legendfontsize=6)
##
savefig(p_statistics, "majorana_similarity_SI_statistics.pdf")