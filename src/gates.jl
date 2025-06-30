
function majorana_exchange(γ1, γ2)
    return sqrt(1im / 2) * (I + γ1 * γ2)
end
function majorana_exchange(parity_operator)
    return sqrt(1im / 2) * (I - 1im * parity_operator)
end

gate_fidelity(g1, g2) = abs(dot(g1, g2)^2 / (dot(g1, g1) * dot(g2, g2)))
gate_fidelity(g1, g2, proj) = abs(dot(g1, proj * g2 * proj)^2 / (dot(g1, proj * g1 * proj) * dot(g2, proj * g2 * proj)))

single_braid_gate_kato(d::Dict) = single_braid_gate_kato(d[:P], d[:ζ], d[:totalparity])
function single_braid_gate_kato(P, ζ, totalparity)
    foldr(*, analytical_gates(P, ζ, totalparity))
end

function analytical_gates(P, ζ, totalparity)
    (; μ, α, β, ν, θ_α, θ_μ) = analytical_components_middle_of_protocol(ζ, totalparity)
    ϕ_μ = π / 4 + θ_μ / 2
    ϕ_α = θ_α / 2
    U_12 = exp(1im * ϕ_μ * P[:M̃, :L] + 1im * ϕ_α * P[:M, :L̃])
    U_23 = exp(1im * π / 4 * (P[:L, :R] + P[:L̃, :R̃])) * exp(-1im * π / 4 * ν * (μ * P[:M̃, :R] + ν * P[:L, :R]) + 1im * π / 4 * α * (β * P[:M, :R̃] - α * P[:L̃, :R̃]))
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

analytical_gate_fidelity(d::Dict) = analytical_gate_fidelity(d[:ζ], d[:totalparity])
function analytical_gate_fidelity(ζ, totalparity)
    (; α, ν) = analytical_components_middle_of_protocol(ζ, totalparity)
    return cos(π / 2 * (1 - abs(α) + abs(ν)))^2
end
single_braid_gate_analytical_angle(d::Dict) = single_braid_gate_analytical_angle(d[:ζ], d[:totalparity])
function single_braid_gate_analytical_angle(ζ, totalparity)
    (; α, ν) = analytical_components_middle_of_protocol(ζ, totalparity)
    return π / 4 * (abs(α) - abs(ν))
end
single_braid_gate_analytical(d::Dict) = single_braid_gate_analytical(d[:P], d[:ζ], d[:totalparity])
function single_braid_gate_analytical(P, ζ, totalparity)
    (; α, ν) = analytical_components_middle_of_protocol(ζ, totalparity)
    return exp(-π / 4 * (abs(α) - abs(ν)) * 1im * P[:L, :R])
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
