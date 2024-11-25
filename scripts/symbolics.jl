using MajoranaBraiding
using QuantumDots
using Majoranas
using LinearAlgebra
using Plots
using OrdinaryDiffEqTsit5
using ProgressMeter
using StaticArrays
using Base.Threads
using Accessors
##
γmb = get_majorana_basis()
N = length(γmb.fermion_basis)
totalparity = 1
use_static_arrays = true
inplace = !use_static_arrays
mtype, vtype = MajoranaBraiding.matrix_vec_types(use_static_arrays, inplace, N)
P = parity_operators(γmb, totalparity, mtype)
# H = ham_with_corrections
## Initial state and identity matrix
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[:M, :M̃] + P[:L, :L̃] + P[:R, :R̃]), 1:1).vectors))))
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
    :γmb => γmb, #Majorana basis
    :u0 => u0, #Initial state. Use U0 for the identity matrix.
    :extra_shifts => [0, 0, 0] #Shifts the three Δ pulses. Given as fractions of T
)

## Solve the system
prob = setup_problem(param_dict)
@time sol = solve(prob[:odeprob], Tsit5(), abstol=1e-6, reltol=1e-6);
plot(sol.t, [1 .- norm(sol(t)) for t in sol.t], label="norm error", xlabel="t")

##
@majoranas γ
Psym = Dict((l1, l2) => 1im * γ[l1] * γ[l2] for l1 in keys(γmb), l2 in keys(γmb))
γbdg = MajoranaWrapper(FermionBdGBasis(1:3), collect(keys(γmb)))
Pbdg = Dict((l1, l2) => 1im * γbdg[l1] * γbdg[l2] for l1 in keys(γmb), l2 in keys(γmb))
##
t = prob[:T] / 2
hamnum = MajoranaBraiding._ham_with_corrections(prob[:ramp], prob[:ϵs], prob[:ζ], prob[:correction], prob[:P], t)
ham = MajoranaBraiding._ham_with_corrections(prob[:ramp], prob[:ϵs], prob[:ζ], prob[:correction], Psym, t)
hambdg = MajoranaBraiding._ham_with_corrections(prob[:ramp], prob[:ϵs], prob[:ζ], prob[:correction], Pbdg, t)
diagham = 1im * prod(diagonal_majoranas(γ, prob[:ramp], t, prob[:ζ]))
hamnum2 = QuantumDots.eval_in_basis(ham, γmb)
hamdiagnum = QuantumDots.eval_in_basis(diagham, γmb)
##
hamnum2[5:8, 5:8] - hamnum
(1im * ham).dict
(1im * diagham).dict
(ham - diagham).dict
##
hamdict = Dict{keytype(ham.dict),Float64}(filter(x -> abs(x[2]) > 1e-3, (1im * ham).dict))
diaghamdict = Dict{keytype(diagham.dict),Float64}((1im * diagham).dict)
for key in keys(diagham.dict)
    get!(hamdict, key, get(hamdict, key, 0))
end
print("coefficients for actual hamiltonian:")
sort(hamdict, by=x -> map(f -> f.label, x.factors))
print("coefficients from the \"diagonal hamiltonian\":")
sort(diaghamdict, by=x -> map(f -> f.label, x.factors))

##
parameters = MajoranaBraiding.analytic_parameters(find_zero_energy_from_analytics(prob[:ζ], prob[:ramp], t, 0.0, 1), prob[:ζ], prob[:ramp], t)

##
(filter(x -> abs(x[2]) > 1e-3, (diagham*ham - ham*diagham).dict))