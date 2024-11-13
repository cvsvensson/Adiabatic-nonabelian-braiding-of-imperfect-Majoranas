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
# Can you write the above function using trace, * and dagger?
#gate_fidelity(g1, g2) = abs(tr(g1' * g2)^2 / (tr(g1' * g1) * tr(g2' * g2)))

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


single_braid_gate_kato(d::Dict) = single_braid_gate_kato(d[:P], d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1))

analytical_gates(d::Dict) = analytical_gates(d[:P], d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1))
function analytical_gates(P, ζ, ramp, T, totalparity)
    θ_α, θ_μ = single_braid_gate_analytical_angles(ζ, ramp, T, totalparity)
    μ = cos(θ_μ)
    α = cos(θ_α)
    ν = sin(θ_μ)
    β = sin(θ_α)
    ϕ_μ = π / 4 - θ_μ/2
    ϕ_α = θ_α/2
    U_12 = exp(+1im * ϕ_μ * P[1, 2] + 1im * ϕ_α * P[0, 4])
    U_23 = exp(+1im * π / 4 * (μ^2 * P[2, 3] + β^2 * P[4, 5])
               + 1im * 1 / 2 * μ * ν * (P[1, 3] - P[1, 2])
               + 1im * 1 / 2 * α * β * (P[0, 5] - P[0, 4]))
    U_31 = exp(-1im * ϕ_μ * P[1, 3] - 1im * ϕ_α * P[0, 5])
    return U_31, U_23, U_12
end
function single_braid_gate_kato(P, ζ, ramp, T, totalparity=1)
    foldr(*, analytical_gates(P, ζ, ramp, T, totalparity))
end

single_braid_gate_lucky_guess(d::Dict) = single_braid_gate_lucky_guess(d[:P], d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1))
function single_braid_gate_lucky_guess(P, ζ, ramp, T, totalparity=1)
    angle = single_braid_gate_analytical_angle(ζ, ramp, T, totalparity)
    U = exp(1im * angle * P[2, 3])
    return U
end

analytical_gate_fidelity(d::Dict) = analytical_gate_fidelity(d[:P], d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1))
function analytical_gate_fidelity(P, ζ, ramp, T, totalparity=1)
    result = find_zero_energy_from_analytics(ζ, ramp, T/2, 0.0, totalparity)
    μ, α, β, ν = groundstate_components(result, ζ, ramp, T/2)
    return cos(π / 4 * (1 - α + ν))^2
end

single_braid_gate_analytical_angle(d::Dict) = single_braid_gate_analytical_angle(d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1))
single_braid_gate_analytical_angles(d::Dict) = single_braid_gate_analytical_angles(d[:ζ], d[:ramp], d[:T], get(d, :totalparity, 1))
function single_braid_gate_analytical_angles(ζ, ramp, T, totalparity=1)
    t = T / 2
    initial = 0.0
    result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity)
    μ, α, β, ν = groundstate_components(result, ζ, ramp, t)
    θ_α = atan(β, α)
    θ_μ = atan(ν, μ)
    return θ_α, θ_μ
end
function single_braid_gate_analytical_angle(ζ, ramp, T, totalparity=1)
    θ_α, θ_μ = single_braid_gate_analytical_angles(ζ, ramp, T, totalparity)
    α = cos(θ_α)
    ν = sin(θ_μ)
    return π / 4 * ((1 - ν) -totalparity * (1 - α) )
end

diagonal_majoranas(d::Dict, t, totalparity=1) = diagonal_majoranas(d[:γ], d[:ramp], t, d[:ζ], totalparity)

function diagonal_majoranas(γ, ramp, t, ζ, totalparity=1)
    result = find_zero_energy_from_analytics(ζ, ramp, t, 0.0, totalparity)
    μ, α, β, ν = groundstate_components(result, ζ, ramp, t)
    Η, Λ = energy_parameters(result, ζ, ramp, t)
    Δs = ramp(t)
    Δtot = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    Δ_23 = √(Δs[2]^2 + Δs[3]^2)
    θ_23 = atan(Δ_23, Δs[1])
    ϕ_23 = atan(Δs[3], Δs[2])

    ρ1 = cos(θ_23)
    ρ2 = sin(θ_23)*cos(ϕ_23)
    ρ3 = sin(θ_23)*sin(ϕ_23)
    
    Δtot *= α*μ - Η*β*ν - Λ*β*μ

    γ_ϕ = cos(ϕ_23) * γ[2] + sin(ϕ_23) * γ[3]
    γ_η = cos(ϕ_23) * γ[4] + sin(ϕ_23) * γ[5]
    γ_θ = cos(θ_23) * γ[1] + sin(θ_23) * γ_ϕ
    γ_Θ_disc = cos(θ_23) * γ_ϕ - sin(θ_23) * γ[1]

    γ1 = α * γ[0] + β * γ_η
    γ2 = μ * γ_ϕ + ν * γ[1]
    #γ2 = μ * γ_θ + ν * γ_Θ_disc    # Generalization of γ2 for Δ_1 > 0

    return γ1, γ2, Δtot
end


function single_braid_gate_fit(ω, P)
    return exp(1im * ω * P[2, 3])
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