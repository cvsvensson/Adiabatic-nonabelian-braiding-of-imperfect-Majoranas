
function majorana_exchange(γ1, γ2)
    return sqrt(1im / 2) * (I + γ1 * γ2)
end
function majorana_exchange(parity_operator)
    return sqrt(1im / 2) * (I - 1im * parity_operator)
end

gate_fidelity(g1, g2) = abs(dot(g1, g2)^2 / (dot(g1, g1) * dot(g2, g2)))
gate_fidelity(g1, g2, proj) = abs(dot(g1, proj * g2 * proj)^2 / (dot(g1, proj * g1 * proj) * dot(g2, proj * g2 * proj)))

single_braid_gate_kato(d::Dict) = single_braid_gate_kato(d[:P], d[:η], d[:totalparity])
function single_braid_gate_kato(P, η, totalparity)
    foldr(*, analytical_gates(P, η, totalparity))
end

function analytical_gates(P, η, totalparity)
    (; μ, α, β, ν, θ_α, θ_μ) = analytical_components_middle_of_protocol(η, totalparity)
    ϕ_μ = π / 4 + θ_μ / 2
    ϕ_α = θ_α / 2
    U_12 = exp(1im * ϕ_μ * P[:M̃, :L] + 1im * ϕ_α * P[:M, :L̃])
    U_23 = exp(1im * π / 4 * (P[:L, :R] - totalparity * P[:L̃, :R̃])) * exp(-1im * π / 4 * ν * (μ * P[:M̃, :R] + ν * P[:L, :R]) + -totalparity * 1im * π / 4 * α * (β * P[:M, :R̃] - α * P[:L̃, :R̃]))
    U_31 = exp(-1im * ϕ_μ * P[:M̃, :R] + totalparity * 1im * ϕ_α * P[:M, :R̃])
    return U_31, U_23, U_12

end

zero_energy_analytic_parameters(d::Dict) = zero_energy_analytic_parameters(d[:η], d[:k], 1, d[:totalparity]; get(d, :opt_kwargs, (;))...)
function zero_energy_analytic_parameters(η::Tuple, k, t, totalparity; kwargs...)
    zero_energy_analytic_parameters(effective_η(η), k, t, totalparity; kwargs...)
end
function zero_energy_analytic_parameters(η, k, t, totalparity; opt_kwargs...)
    initial = 0.0
    result = find_zero_energy_from_analytics(η, k, t, initial, totalparity; opt_kwargs...)
    return analytic_parameters(result, η, k, t)
end

analytical_gate_fidelity(d::Dict) = analytical_gate_fidelity(d[:η], d[:totalparity])
analytical_gate_fidelity(η::Tuple, totalparity) = analytical_gate_fidelity(effective_η(η), totalparity)
function analytical_gate_fidelity(η, totalparity)
    # (; α, ν) = analytical_components_middle_of_protocol(η, totalparity)
    # return sin(π / 2 * (-totalparity * α + ν))^2
    return sin(π / 2 * (1 - η^2)^(3 / 2) / (1 - η^6)^(1 / 2))^2
end
single_braid_gate_analytical(d::Dict) = single_braid_gate_analytical(d[:P], d[:η], d[:totalparity])
function single_braid_gate_analytical(P, η, totalparity)
    (; α, ν) = analytical_components_middle_of_protocol(η, totalparity)
    return exp(π / 4 * (α + ν) * 1im * P[:L, :R])
end


diagonal_majoranas(d::Dict, t, λ) = diagonal_majoranas(d[:η], d[:k], t, λ)
diagonal_majoranas(d::Dict, t) = diagonal_majoranas_at_zero_energy(d[:η], d[:k], t, d[:totalparity])

function diagonal_majoranas_at_zero_energy(η, k, t, totalparity)
    λ = find_zero_energy_from_analytics(η, k, t, 0.0, totalparity)
    diagonal_majoranas(η, k, t, λ, totalparity)
end
function diagonal_majoranas(η, k, t, λ, protocol_parity)
    γ = majoranas(majorana_hilbert_space(MajoranaLabels, ParityConservation()))
    (; ηtilde, λtilde, μ, α, β, ν, θ_α, θ_μ, θ, ϕ) = analytic_parameters(λ, η, k, t)
    sθ, cθ = sincos(θ)
    sϕ, cϕ = sincos(ϕ)

    γΔ = cθ * γ[:M̃] + sθ * cϕ * γ[:L] + sθ * sϕ * γ[:R]

    γθprime = -sθ * γ[:M̃] + cθ * cϕ * γ[:L] + cθ * sϕ * γ[:R]
    γϕprime = -sϕ * γ[:L] + cϕ * γ[:R]
    γη = cϕ * γ[:L̃] - protocol_parity * sϕ * γ[:R̃]
    γηprime = -sϕ * γ[:L̃] - protocol_parity * cϕ * γ[:R̃]

    γ1D = α * γ[:M] + β * γη
    γΔD = μ * γΔ + ν * γθprime
    γηD = α * γη - β * γ[:M]
    γθDprime = μ * γθprime - ν * γΔ

    return γ1D, γΔD, γηD, γθDprime, γϕprime, γηprime
end

@testitem "Diagonal majoranas" begin
    using LinearAlgebra
    using FermionicHilbertSpaces
    γ = majoranas(majorana_hilbert_space(MajoranaBraiding.MajoranaLabels, ParityConservation()))
    T = 1e4
    η = 0.5
    k = 1e1
    ts = range(0, 2, 100)
    parity_operator = 1im * prod(values(γ))
    λ = 0
    protocol_parity = 1
    γdiag = diagonal_majoranas(η, k, 1 / 3, λ, protocol_parity)
    parity_operator_diag = 1im * prod(γdiag)
    @test parity_operator ≈ protocol_parity * parity_operator_diag
    @test parity_operator ≈ -protocol_parity * 1im * prod(diagonal_majoranas(η, k, 1 / 3, λ, -protocol_parity))

    @test all(norm(y^2 - I) < 1e-10 for y in γdiag)
    ## test CAR 
    for (y1, y2) in Base.product(values(γ), values(γ))
        @test norm(y1 * y2 + y2 * y1 - 2I * (y1 == y2)) < 1e-10
    end
    for (y1, y2) in Base.product(γdiag, γdiag)
        @test norm(y1 * y2 + y2 * y1 - 2I * (y1 == y2)) < 1e-10
    end

    #Non-zero lambda
    λ = 0.3
    γdiag = diagonal_majoranas(η, k, 1 / 3, λ, protocol_parity)
    parity_operator_diag = 1im * prod(γdiag)
    @test parity_operator ≈ protocol_parity * parity_operator_diag
    @test parity_operator ≈ -protocol_parity * 1im * prod(diagonal_majoranas(η, k, 1 / 3, λ, -protocol_parity))


    @test all(norm(y^2 - I) < 1e-10 for y in γdiag)
    ## test CAR 
    for (y1, y2) in Base.product(values(γ), values(γ))
        @test norm(y1 * y2 + y2 * y1 - 2I * (y1 == y2)) < 1e-10
    end
    for (y1, y2) in Base.product(γdiag, γdiag)
        @test norm(y1 * y2 + y2 * y1 - 2I * (y1 == y2)) < 1e-10
    end

    # Check that the hamiltonian is the same
    param_dict = Dict(
        :η => 0.2,
        :T => 5e2,
        :k => 1e1,
        :steps => 200,
        :correction => SimpleCorrection(sin),
        :interpolate_corrected_hamiltonian => true,
        :initial => I,
        :totalparity => -1
    )
    prob = setup_problem(param_dict)
    function diagham(d, t)
        subinds = d[:totalparity] == 1 ? (5:8) : (1:4)
        # subinds = (1:4)
        γ1D, γΔD, γηD, γθDprime, γϕprime, γηprime = diagonal_majoranas(d[:η], d[:k], t, d[:correction].scaling(t), d[:totalparity])
        (; ε, Δtilde, λ) = MajoranaBraiding.analytic_parameters(d[:correction].scaling(t), d[:η], d[:k], t)
        1im * (Δtilde*γ1D*γΔD+ε*γηD*γθDprime+λ*γϕprime*γηprime)[subinds, subinds]
    end
    @test all(prob[:H](prob[:p], t) / prob[:T] ≈ diagham(prob, t) for t in range(0, 2, 100))

    param_dict[:totalparity] = -param_dict[:totalparity]
    prob = setup_problem(param_dict)
    @test all(prob[:H](prob[:p], t) / prob[:T] ≈ diagham(prob, t) for t in range(0, 2, 100))
end
