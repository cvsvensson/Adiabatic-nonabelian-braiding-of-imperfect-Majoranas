using MajoranaBraiding
using QuantumDots
using Majoranas
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using ProgressMeter
using StaticArrays
using Base.Threads
using Roots

##
nbr_of_majoranas = 6
N = nbr_of_majoranas ÷ 2
majorana_labels = 0:5
γ = SingleParticleMajoranaBasis(nbr_of_majoranas, majorana_labels)
parity = 1
use_static_arrays = true
inplace = !use_static_arrays
mtype, vtype = MajoranaBraiding.matrix_vec_types(use_static_arrays, inplace, N)
P = parity_operators(γ, parity, mtype)
H = ham_with_corrections
## Parameters
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[:M, :M̃] + P[:L, :L̃] + P[:R, :R̃]), 1:1).vectors))))
ρmax = 1
T = 2e3 / ρmax
k = 1e1
ρmin = 1e-6 * ρmax
η = 7e-1
ηs = (η, η, η) # Unwanted Majorana contributions within each island ordered as η01, η24, η35
tspan = (0.0, 2T)
ts = range(0, tspan[2], 1000)
ramp = RampProtocol([2, 1, 1 / 2] .* ρmin, [2, 4, 1] .* ρmax, T, k)

## 
simplecorr = optimized_simple_correction(H, (ramp, ηs, P), ts)
independentsimplecorr = optimized_independent_simple_correction(H, (ramp, ηs, P), ts)
analyticsimplecorr = analytical_exact_simple_correction(η, ramp, ts)
remove_labels = [[0, 1], [0, 2], [0, 3]]
constrained_basis = MajoranaBraiding.remove_from_basis(remove_labels, P)

corrections = [simplecorr, independentsimplecorr, analyticsimplecorr]
##

for (n, corr) in enumerate(corrections)
    p = (ramp, ηs, corr, P)
    pl = visualize_protocol(H, ramp, ηs, corr, P, ts)
    annotate!([(:left, :top, "Correction: $n")])
    display(pl)
    # deltas = stack([ramp(t) for t in ts])'
    # delta_plot = plot(ts, deltas, label=["ρ01" "ρ02" "ρ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)
    # spectrum = stack([eigvals(H(p, t)) for t in ts])'
    # plot(plot(ts, mapslices(v -> v[2:end] .- v[1], spectrum, dims=2), ls=[:solid :dash :dot], title="$n: Eᵢ-E₀", labels=[1, 2, 3]', yscale=:log10, ylims=(1e-16, 1e1)), delta_plot, layout=(2, 1), lw=2, frame=:box) |> display
end