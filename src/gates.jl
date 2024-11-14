function single_qubit_gates(zero, one)
    @assert norm(zero) ≈ norm(one) ≈ 1 "basis states are not normalized"
    @assert abs(dot(zero, one)) < 1e-14 "basis states are not orthogonal"
    gates = Dict(:X => one * zero' + zero * one',
        :Y => -im * one * zero' + im * zero * one',
        :Z => zero * zero' - one * one',
        :I => zero * zero' + one * one')
    gates[:H] = sqrt(1 / 2) * (gates[:X] + gates[:Z])
    return gates
end

@testitem "Single gates" begin
    using LinearAlgebra
    basis = svd(rand(10, 2)).U
    v0, v1 = eachcol(basis) #random orthonormal states
    gates = MajoranaBraiding.single_qubit_gates(v0, v1)
    @test gates[:X] * v0 ≈ v1
    @test gates[:Y] * v0 ≈ -im * v1
    @test gates[:Z] * v0 ≈ v0
    @test gates[:Z] * v1 ≈ -v1
    @test gates[:H] * v0 ≈ (v0 + v1) / sqrt(2)
    @test basis' * gates[:X]' * gates[:X] * basis ≈ I

    @test gate_fidelity(gates[:X], gates[:X]) ≈ 1
    @test gate_fidelity(gates[:X], gates[:Y]) < 1e-14

    @test all(MajoranaBraiding.gate_overlaps(gates[S], gates)[S] ≈ 1 for S in (:X, :Y, :Z, :I))
end

function majorana_exchange(γ1, γ2)
    return sqrt(1im / 2) * (I + γ1 * γ2)
end
function majorana_exchange(parity_operator)
    return sqrt(1im / 2) * (I - 1im * parity_operator)
end


majorana_braid(γ1, γ2) = majorana_exchange(γ1, γ2)^2
majorana_braid(parity_operator) = majorana_exchange(parity_operator)^2

function gate_overlaps(gate, gates::Dict)
    Dict(k => tr(gate * v) / 2 for (k, v) in pairs(gates))
end

gate_fidelity(g1, g2) = abs(dot(g1, g2)^2 / (dot(g1, g1) * dot(g2, g2)))

@testitem "majorana_exchange" begin
    using LinearAlgebra, QuantumDots
    c = FermionBasis(1:2, qn=QuantumDots.parity)
    majorana_labels = 0:3
    γ = MajoranaWrapper(c, majorana_labels)

    vacuum_state_ind = c.symmetry.focktoinddict[0]
    vacuum_state = zeros(2^length(keys(c)))
    vacuum_state[vacuum_state_ind] = 1
    one_state = c[2]' * c[1]' * vacuum_state
    gates = single_qubit_gates(vacuum_state, one_state)
    basis = hcat(vacuum_state, one_state)

    B01 = majorana_exchange(γ[0], γ[1])
    @test B01' * B01 ≈ I
    @test B01 ≈ majorana_exchange(1im * γ[0] * γ[1])
    @test majorana_braid(γ[0], γ[1]) ≈ majorana_braid(1im * γ[0] * γ[1])

    @test majorana_exchange(γ[0], γ[1])^2 ≈ majorana_braid(γ[0], γ[1])

    majorana_braid(γ[1], γ[2])' * majorana_braid(γ[1], γ[2]) ≈ I
    @test gate_overlaps(majorana_braid(γ[1], γ[2]), gates)[:X] ≈ -1


    @test gate_overlaps(majorana_braid(γ[1], γ[3]), gates)[:Y] ≈ -1
end


# analytical_protocol_gate(d::Dict) = analytical_protocol_gate(d[:P], d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1); get(d, :opt_kwargs, (;))...)
# function analytical_protocol_gate(P, ζ, ramp, T, totalparity; opt_kwargs...)
#     (; α, ν) = zero_energy_analytic_parameters(ζ, ramp, T, totalparity; opt_kwargs...)
#     return exp(1im * π / 4 * ((1 - α) * P[:L, :R] - (1 - ν) * P[:L̃, :R̃]))
# end
single_braid_gate_kato(d::Dict) = single_braid_gate_kato(d[:P], d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1); get(d, :opt_kwargs, (;))...)
function single_braid_gate_kato(P, ζ, ramp, T, totalparity=1; opt_kwargs...)
    foldr(*, analytical_gates(P, ζ, ramp, T, totalparity; opt_kwargs...))
end

single_braid_gate_lucky_guess(d::Dict) = single_braid_gate_lucky_guess(d[:P], d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1); get(d, :opt_kwargs, (;))...)
function single_braid_gate_lucky_guess(P, ζ, ramp, T, totalparity=1; opt_kwargs...)
    (; α, ν) = zero_energy_analytic_parameters(ζ, ramp, T, totalparity; opt_kwargs...)
    # α = cos(θ_α)
    # ν = sin(θ_μ)
    # U = exp(1im * π / 4 * ((1 - α) * P[:L, :R] + totalparity * (1 + ν) * P[:L̃, :R̃]))
    exp(1im * π / 4 * (α * P[:L, :R] + ν * P[:L̃, :R̃]))
    # return U
end
function analytical_gates(P, ζ, ramp, T, totalparity; opt_kwargs...)
    (; μ, α, β, ν, θ_α, θ_μ) = zero_energy_analytic_parameters(ζ, ramp, T, totalparity; opt_kwargs...)
    ϕ_μ = π / 4 - θ_μ / 2
    ϕ_α = θ_α / 2
    # U_12 = exp(1im * ϕ_μ * P[1, 2] + 1im * ϕ_α * P[0, 4])
    # U_23 = exp(1im * π / 4 * (μ^2 * P[2, 3] + β^2 * P[4, 5])
    #            + 1im * 1 / 2 * μ * ν * (P[1, 3] - P[1, 2])
    #            + 1im * 1 / 2 * α * β * (P[0, 5] - P[0, 4]))
    # U_31 = exp(-1im * ϕ_μ * P[1, 3] - 1im * ϕ_α * P[0, 5])
    # label_replacements = [0 => :M, 1 => :M̃, 2=>:L, 3=>:R, 4=>:L̃, 5=>:R̃]
    U_12 = exp(1im * ϕ_μ * P[:M̃, :L] + 1im * ϕ_α * P[:M, :L̃])
    U_23 = exp(1im * π / 4 * (μ^2 * P[:L, :R] + β^2 * P[:L̃, :R̃])
               + 1im * 1 / 2 * μ * ν * (P[:M̃, :R] - P[:M̃, :L])
               + 1im * 1 / 2 * α * β * (P[:M, :R̃] - P[:M, :L̃]))
    U_31 = exp(-1im * ϕ_μ * P[:M̃, :R] - 1im * ϕ_α * P[:M, :R̃])
    return U_31, U_23, U_12
end

zero_energy_analytic_parameters(d::Dict) = zero_energy_analytic_parameters(d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1); get(d, :opt_kwargs, (;))...)
function zero_energy_analytic_parameters(ζ, ramp, T, totalparity=1; opt_kwargs...)
    t = T / 2
    initial = 0
    result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity; opt_kwargs...)
    return analytic_parameters(result, ζ, ramp, t)
end

# single_braid_gate_analytical_angles(d::Dict) = single_braid_gate_analytical_angles(d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1))
# function single_braid_gate_analytical_angles(ζ, ramp, T, totalparity=1)
#     t = T / 2
#     initial = 0.0
#     result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity)
#     H, Λ, μ, α, β, ν, θ_α, θ_μ = analytic_parameters(result, ζ, ramp, t)
#     return α, ν
# end
# function single_braid_gate_analytical_angle(ζ, ramp, T, totalparity=1)
#     α, ν = single_braid_gate_analytical_angles(ζ, ramp, T, totalparity)
#     # α = cos(θ_α)^1
#     # ν = sin(θ_μ)^1
#     return π / 4 * (α - ν)
# end


diagonal_majoranas(d::Dict, t, totalparity=1) = diagonal_majoranas(d[:γ], d[:ramp], t, d[:ζ], totalparity)

function diagonal_majoranas(γ, ramp, t, ζ, totalparity=1)
    result = find_zero_energy_from_analytics(ζ, ramp, t, 0.0, totalparity)
    μ, α, β, ν, θ_α, θ_μ = groundstate_components(result, ζ, ramp, t)
    Δs = ramp(t)
    Δtot = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    ρ2 = Δs[2] / Δtot
    ρ3 = Δs[3] / Δtot

    γ1 = α * γ[:M] + β * (ρ2 * γ[:L̃] + ρ3 * γ[:R̃])
    γ2 = μ * (ρ2 * γ[:L] + ρ3 * γ[:R]) + ν * γ[:M̃]

    return γ1, γ2, Δtot
end


function single_braid_gate_fit(ω, P)
    return exp(1im * ω * P[:L, :R])
end

function braid_gate_prediction(gate, ω, P)
    prediction = single_braid_gate_fit(ω, P)

    proj = Diagonal([0, 1, 1, 0])
    single_braid_fidelity = gate_fidelity(proj * prediction * proj, proj * gate * proj)
    return single_braid_fidelity
end

function braid_gate_best_angle(gate, P)
    ω = optimize(ω -> 1 - braid_gate_prediction(gate, ω, P), 0.0, π / 2).minimizer
    return ω, braid_gate_prediction(gate, ω, P)
end