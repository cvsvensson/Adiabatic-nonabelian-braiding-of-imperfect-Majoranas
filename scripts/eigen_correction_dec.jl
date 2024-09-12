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

function corr_matrix(t, p, H, basis, alg=Majoranas.WM_BACKSLASH())
    ham_at_t = H(p, t)
    return MajoranaBraiding.full_energy_correction_term(ham_at_t, basis; alg)
end
function error_ham(t, p)
    return MajoranaBraiding._error_ham(ramp, t, ζs, last(p))
end
function projection_op(t, p, H)
    ham_at_t = H(p, t)
    _, vecs = eigen(Hermitian(ham_at_t))
    return vecs[:, 1:2]
end
function corr_err_sum(Hcorr, Herr, proj)
    return proj'*(Hcorr + Herr)*proj
end
function corr_err_sum_compare_to_diag(corr_err_sum)
    λ = corr_err_sum[1,1]
    return norm(corr_err_sum - λ*I)
end

##
nbr_of_majoranas = 6
N = nbr_of_majoranas ÷ 2
majorana_labels = 0:5
γ = SingleParticleMajoranaBasis(nbr_of_majoranas, majorana_labels)
parity = 1
use_static_arrays = true
inplace = !use_static_arrays
mtype, vtype = MajoranaBraiding.matrix_vec_types(use_static_arrays, inplace, N)
P = parity_operators(γ, parity, mtype) # should we remove identity?
#=Pold = MajoranaBraiding.parity_operators_old(nbr_of_majoranas, majorana_labels, mtype)=#

H = ham_with_corrections
## Parameters
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
dt = 5
ts = range(0, tspan[2], Int(tspan[2] / dt))
ramp = RampProtocol([1, 1, 1] .* Δmin, [1, 1, 1] .* Δmax, T, k)
corr = MajoranaBraiding.NoCorrection()
p = (ramp, ϵs, ζs, corr, P)
remove_labels = [[2,4],[2,5]]
constrained_basis = MajoranaBraiding.remove_from_basis(remove_labels, P)

full_corr_vec = map(t->corr_matrix(t, p, H, constrained_basis), ts)
full_corr_comp_vec = map(mat->Majoranas.matrix_to_dict(constrained_basis, mat), full_corr_vec)
error_hams = map(t->error_ham(t, p), ts)
projs = map(t->projection_op(t, p, H), ts)
corr_err_sums = map((Hcorr, Herr, proj)->corr_err_sum(Hcorr, Herr, proj), full_corr_vec, error_hams, projs)
comp_to_id = map(comp_err_sum->corr_err_sum_compare_to_diag(comp_err_sum), corr_err_sums)

comp_to_id_plot = plot(ts, comp_to_id, title="|P(Herr+Hcorr)P-λI|")
deltas = stack([ramp(t) for t in ts])'
delta_plot = plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)
labs = [key for key in keys(constrained_basis)]
labs = [[0,1],[0,2],[0,3],[1,4], [1,5]]
chosen_corr_comp_vec = mapreduce(corr_comps -> [real(corr_comps[lab]) for lab in labs]', vcat, full_corr_comp_vec)
comp_plot = plot(ts, chosen_corr_comp_vec, labels=permutedims(labs))
plot(delta_plot, comp_to_id_plot, comp_plot, layout=(3,1))


#=ham_at_t = H(p, T/3)=#
#=eig_corr = MajoranaBraiding.full_energy_correction_term(ham_at_t, last(p))=#
#=vals, vecs = eigen(Hermitian(ham_at_t))=#
#=δE = (vals[2] - vals[1]) / 2=#
#=# push the lowest energy states δE closer together=#
#=remove_labels = [[2,5], [3,4], [1,4],[1,5]]=#
#=constrained_basis = MajoranaBraiding.remove_from_basis(remove_labels, P)=#
#=weak_ham_prob = WeakMajoranaProblem(constrained_basis, vecs, nothing, [nothing, nothing, nothing, δE]) # should we really restrict identity in gs?=#
#=sol = solve(weak_ham_prob, Majoranas.WM_BACKSLASH())=#
#=con = weak_ham_prob.constraints=#
