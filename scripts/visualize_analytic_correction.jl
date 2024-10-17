using MajoranaBraiding
using QuantumDots
using Majoranas
using LinearAlgebra
using Plots
using ProgressMeter

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
u0 = vtype(collect(first(eachcol(eigen(Hermitian(P[0, 1] + P[2, 4] + P[3, 5]), 1:1).vectors))))
Δmax = 1
T = 2e3 / Δmax
k = 1e1
Δmin = 1e-6 * Δmax
ϵs = (0.0, 0.0, 0.0) # Energy overlaps between Majoranas ordered as ϵ01, ϵ24, ϵ35
ζ = 7e-1
ζs = (ζ, ζ, ζ) # Unwanted Majorana contributions within each island ordered as ζ01, ζ24, ζ35
tspan = (0.0, 2T)
# Take ts with one step per time unit
dt = 2
ts = range(0, tspan[2], Int(tspan[2] / dt))
ramp = RampProtocol([2, 1, 1] .* Δmin, [1, 1, 1] .* Δmax, T, k)
p = (ramp, ϵs, ζs, corr, P)

## Check the energy spliting in dependence of λ
t = 2 * T / 4
λ_array = range(-1, 1, length=100)
energy_split_array = [MajoranaBraiding.energy_splitting(λ, ζ, ramp, t, parity) for λ in λ_array]
plot(λ_array, energy_split_array, label="Energy split", xlabel="λ", ylabel="Energy split", lw=2, frame=:box)

## Visualize the correctionηt = T/2
result = find_zero_energy_from_analytics(ζ, ramp, t, parity)
μ, α, β, ν = MajoranaBraiding.groundstate_components(result, ζ, ramp, t)

ζ_array = range(1e-4, 1 - 1e-3, length=100)
ground_state_array = [MajoranaBraiding.groundstate_components(find_zero_energy_from_analytics(ζ, ramp, t, parity), ζ^2, ramp, t) for ζ in ζ_array]
# Plot all the components of the ground state as a function of ζ
labels = ["μ", "α", "β", "ν"]
for (idx, component) in enumerate(ground_state_array[1])
    if idx == 1
        plot(ζ_array, [c[idx] for c in ground_state_array], label=labels[idx], xlabel="ζ", ylabel="Component", lw=2, frame=:box)
    else
        plot!(ζ_array, [c[idx] for c in ground_state_array], label=labels[idx], xlabel="ζ", ylabel="Component", lw=2, frame=:box)
    end
end
# Show the plot
plot!()