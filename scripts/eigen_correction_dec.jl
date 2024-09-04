using MajoranaBraiding
using QuantumDots
using Majoranas
using LinearAlgebra
using Plots
using OrdinaryDiffEq
using ProgressMeter
using StaticArrays
using Base.Threads
# using TaylorSeries
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
#=Pold = MajoranaBraiding.parity_operators_old(nbr_of_majoranas, majorana_labels, mtype)=#

H = ham_with_corrections
P
## Parameters
P
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[0, 1] + P[2, 4] + P[3, 5]), 1:1).vectors))))
Δmax = 1
T = 1e3 / Δmax
k = 1e1
Δmin = 1e-6 * Δmax
ϵs = (0.0, 0.0, 0.0) # Energy overlaps between Majoranas ordered as ϵ01, ϵ24, ϵ35
ζ = 9.9e-1
ζs = (ζ, ζ, ζ) # Unwanted Majorana contributions within each island ordered as ζ01, ζ24, ζ35
tspan = (0.0, 2T)
# Take ts with one step per time unit
dt = 100
ts = range(0, tspan[2], Int(tspan[2] / dt))
ramp = RampProtocol([1, 1, 1] .* Δmin, [1, 1, 1] .* Δmax, T, k)
corr = MajoranaBraiding.NoCorrection()
p = (ramp, ϵs, ζs, corr, P)
#=filter(>(1e-3)∘abs, components)=#

function corr_components(t, p, H)
    ham_at_t = H(p, t)
    eig_corr = MajoranaBraiding.full_energy_correction_term(ham_at_t, last(p))
    return Majoranas.matrix_to_dict(P, eig_corr)
end
full_corr_comp_vec = map(t->corr_components(t, p, H), ts)

labs = [[1, 4], [1, 5]]
chosen_corr_comp_vec = mapreduce(corr_comps -> [real(corr_comps[lab]) for lab in labs]', vcat, full_corr_comp_vec)
comp_plot = plot(ts, chosen_corr_comp_vec, labels=permutedims(labs))
deltas = stack([ramp(t) for t in ts])'
delta_plot = plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)
plot(delta_plot, comp_plot, layout=(2,1))

