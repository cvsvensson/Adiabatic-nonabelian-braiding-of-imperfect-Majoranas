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
gate_fidelity(g1, g2, proj) = abs(dot(g1, proj * g2 * proj)^2 / (dot(g1, proj * g1 * proj) * dot(g2, proj * g2 * proj)))

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
single_braid_gate_kato(d::Dict) = single_braid_gate_kato(d[:P], d[:ζ], d[:ramp], d[:T], d[:totalparity]; get(d, :opt_kwargs, (;))...)
function single_braid_gate_kato(P, ζ, ramp, T, totalparity; opt_kwargs...)
    foldr(*, analytical_gates(P, ζ, ramp, T, totalparity; opt_kwargs...))
end

single_braid_gate_analytical(d::Dict) = single_braid_gate_analytical(d[:P], d[:ζ], d[:ramp], d[:totalparity])
single_braid_gate_analytical(P, ζ::Tuple, ramp, totalparity) = single_braid_gate_analytical(P, effective_ζ_by_η(ζ), ramp, totalparity)
function single_braid_gate_analytical(P, ζ, ramp, totalparity; opt_kwargs...)
    # λ = find_zero_energy_from_analytics_midpoint(ζ, ramp, totalparity; opt_kwargs...)
    # η = ζ^2
    # θ_μ = -1 / 2 * atan(2 * λ * η / (1 + λ^2 - η^2))
    # ν = sin(θ_μ)
    # θ_α = -1 * atan(-η * tan(θ_μ) + λ)
    # α = cos(θ_α)
    (; α, ν) = analytic_parameters_midpoint(ζ, totalparity)
    return exp(π / 4 * (1 + ν) * 1im * P[:L, :R]) * exp(π / 4 * (1 - α) * 1im * P[:L̃, :R̃])
end
function analytical_gates(P, ζ, ramp, T, totalparity; opt_kwargs...)
    (; μ, α, β, ν, θ_α, θ_μ) = zero_energy_analytic_parameters(ζ, ramp, T / 2, totalparity; opt_kwargs...)
    ϕ_μ = π / 4 + θ_μ / 2
    ϕ_α = θ_α / 2
    U_12 = exp(1im * ϕ_μ * P[:M̃, :L] + 1im * ϕ_α * P[:M, :L̃])
    U_23 = exp(1im * π / 4 * (P[:L, :R] + P[:L̃, :R̃])) * exp(-1im * π / 4 * ν * (μ * P[:M̃, :R] + ν * P[:L, :R]) + 1im * π / 4 * α * (β * P[:M, :R̃] - α * P[:L̃, :R̃]))
    # U_23 = exp(+1im * π / 4 * (P[2, 3] + P[4, 5])) * exp(1im * π / 4 * ν * (μ * P[1, 3] - ν * P[2, 3]) + 1im * π / 4 * α * (β * P[0, 5] - α * P[4, 5]))
    # label_replacements = [0 => :M, 1 => :M̃, 2=>:L, 3=>:R, 4=>:L̃, 5=>:R̃]
    U_31 = exp(-1im * ϕ_μ * P[:M̃, :R] - 1im * ϕ_α * P[:M, :R̃])
    return U_31, U_23, U_12

end

zero_energy_analytic_parameters(d::Dict) = zero_energy_analytic_parameters(d[:ζ], d[:ramp], d[:T], d[:totalparity]; get(d, :opt_kwargs, (;))...)
function zero_energy_analytic_parameters(ζ::Tuple, ramp, t, totalparity; kwargs...)
    zero_energy_analytic_parameters(effective_ζ_by_η(ζ), ramp, t, totalparity; kwargs...)
end
function zero_energy_analytic_parameters(ζ, ramp, t, totalparity; opt_kwargs...)
    initial = 0.0
    result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity; opt_kwargs...)
    return analytic_parameters(result, ζ, ramp, t)
end

analytical_gate_fidelity(d::Dict) = analytical_gate_fidelity(d[:ζ], d[:ramp], d[:T], d[:totalparity])
function analytical_gate_fidelity(ζ, ramp, T, totalparity; opt_kwargs...)
    (; μ, α, β, ν, θ_α, θ_μ) = zero_energy_analytic_parameters(ζ, ramp, T / 2, totalparity; opt_kwargs...)
    return cos(π / 4 * (1 - α - ν))^2
end

single_braid_gate_analytical_angles(d::Dict) = single_braid_gate_analytical_angles(d[:ζ], d[:ramp], d[:T], d[:totalparity])
function single_braid_gate_analytical_angles(ζ, ramp, T, totalparity)
    # initial = 0.0
    result = find_zero_energy_from_analytics_midpoint(ζ, ramp, totalparity)
    (; ηtilde, λtilde, μ, α, β, ν, θ_α, θ_μ) = analytic_parameters(result, ζ, ramp, T / 2)
    return α, ν
end
single_braid_gate_analytical_angle(d::Dict) = single_braid_gate_analytical_angle(d[:ζ], d[:ramp], d[:T], d[:totalparity])
function single_braid_gate_analytical_angle(ζ, ramp, T, totalparity)
    α, ν = single_braid_gate_analytical_angles(ζ, ramp, T, totalparity)
    return π / 4 * ((1 + ν) - totalparity * (1 - α))
end


diagonal_majoranas(d::Dict, t, λ) = diagonal_majoranas(d[:γ], d[:ramp], t, d[:ζ], λ)
diagonal_majoranas(d::Dict, t) = diagonal_majoranas_at_zero_energy(d[:γ], d[:ramp], t, d[:ζ], d[:totalparity])

function diagonal_majoranas_at_zero_energy(γ, ramp, t, ζ, totalparity)
    λ = find_zero_energy_from_analytics(ζ, ramp, t, 0.0, totalparity)
    diagonal_majoranas(γ, ramp, t, ζ, λ)
end
function diagonal_majoranas(γ, ramp, t, ζ, λ)
    (; ηtilde, λtilde, μ, α, β, ν, θ_α, θ_μ, θ, ϕ) = analytic_parameters(λ, ζ, ramp, t)
    sθ, cθ = sincos(θ)
    sϕ, cϕ = sincos(ϕ)

    γΔ = cθ * γ[:M̃] + sθ * cϕ * γ[:L] + sθ * sϕ * γ[:R]

    γθprime = -sθ * γ[:M̃] + cθ * cϕ * γ[:L] + cθ * sϕ * γ[:R]
    γϕprime = -sϕ * γ[:L] + cϕ * γ[:R]
    γη = cϕ * γ[:L̃] + sϕ * γ[:R̃]
    γηprime = -sϕ * γ[:L̃] + cϕ * γ[:R̃]

    γ1D = α * γ[:M] + β * γη
    γΔD = μ * γΔ + ν * γθprime
    γηD = α * γη - β * γ[:M]
    γθDprime = μ * γθprime - ν * γΔ

    return γ1D, γΔD, γηD, γθDprime, γϕprime, γηprime
    # return γ1, γ2, Δtot
    # Sectionwise definition of the diagonal Majoranas
    # if 0 <= t % T < T / 3
    #     γ_1 = γ[:M]
    #     γ_η = γ[:L̃]
    #     γ_Δ = ρ1 * γ[:M̃] + ρ2 * γ[:L]
    #     γ_Δ_disc = ρ1 * γ[:L] - ρ2 * γ[:M̃]
    #     γ1 = α * γ_1 + β * γ_η
    #     γ2 = μ * γ_Δ + ν * γ_Δ_disc
    # elseif T / 3 <= t % T < 2 * T / 3
    #     γ_1 = γ[:M]
    #     γ_η = ρ2 * γ[:L̃] + ρ3 * γ[:R̃]
    #     γ_1̃ = γ[:M̃]
    #     γ_Δ = ρ2 * γ[:L] + ρ3 * γ[:R]
    #     γ1 = α * γ_1 + β * γ_η
    #     γ2 = μ * γ_Δ + ν * γ_1̃
    # else
    #     γ_1 = γ[:M]
    #     γ_η = γ[:R̃]
    #     γ_Δ = ρ1 * γ[:M̃] + ρ3 * γ[:R]
    #     γ_Δ_disc = ρ1 * γ[:R] - ρ3 * γ[:M̃]
    #     γ1 = α * γ_1 + β * γ_η
    #     γ2 = μ * γ_Δ + ν * γ_Δ_disc
    # end

end

@testitem "Diagonal majoranas" begin
    using LinearAlgebra
    γ = get_majorana_basis()
    T = 1e4
    ramp = RampProtocol(0, 1e6, T, 1e1)
    ζ = 0.5
    ts = range(0, 2T, 100)
    parity_operator = 1im * prod(γ)
    λ = 0
    γdiag = diagonal_majoranas(γ, ramp, T / 3, ζ, λ)
    parity_operator_diag = 1im * prod(γdiag)
    @test parity_operator ≈ parity_operator_diag

    @test all(norm(y^2 - I) < 1e-10 for y in γdiag)
    ## test CAR 
    for (y1, y2) in Base.product(γ, γ)
        @test norm(y1 * y2 + y2 * y1 - 2I * (y1 == y2)) < 1e-10
    end
    for (y1, y2) in Base.product(γdiag, γdiag)
        @test norm(y1 * y2 + y2 * y1 - 2I * (y1 == y2)) < 1e-10
    end

    #Non-zero lambda
    λ = 0.3
    γdiag = diagonal_majoranas(γ, ramp, T / 3, ζ, λ)
    parity_operator_diag = 1im * prod(γdiag)
    @test parity_operator ≈ parity_operator_diag

    @test all(norm(y^2 - I) < 1e-10 for y in γdiag)
    ## test CAR 
    for (y1, y2) in Base.product(γ, γ)
        @test norm(y1 * y2 + y2 * y1 - 2I * (y1 == y2)) < 1e-10
    end
    for (y1, y2) in Base.product(γdiag, γdiag)
        @test norm(y1 * y2 + y2 * y1 - 2I * (y1 == y2)) < 1e-10
    end
end
