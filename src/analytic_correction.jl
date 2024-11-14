struct InterpolatedExactSimpleCorrection{T} <: AbstractCorrection
    totalparity::T
end
InterpolatedExactSimpleCorrection() = InterpolatedExactSimpleCorrection(1)

function setup_correction(corr::InterpolatedExactSimpleCorrection, d::Dict)
    ζ = d[:ζ]
    ramp = d[:ramp]
    ts = d[:ts]
    return analytical_exact_simple_correction(ζ, ramp, ts, corr.totalparity)
end
function analytical_exact_simple_correction(ζ, ramp, ts, totalparity=1; opt_kwargs...)
    results = Float64[]
    for t in ts
        # Find roots of the energy split function
        initial = length(results) > 0 ? results[end] : 0.0
        result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity; opt_kwargs...)
        push!(results, result)
    end
    return SimpleCorrection(linear_interpolation(ts, results, extrapolation_bc=Periodic()))
end

function find_zero_energy_from_analytics(ζ, ramp, t, initial=0.0, totalparity=1; kwargs...)
    result = find_zero(x -> energy_splitting(x, ζ, ramp, t, totalparity), initial; kwargs...)
    return result
end

""" 
    energy_splitting(x, ζ, ramp, t, totalparity=1)

Calculate (analytically) the energy splitting between the two lowest energy levels of the system. Works only when all ζs are the same?
"""
function energy_splitting(x, ζ, ramp, t, totalparity=1)
    (; H, Λ, μ, α, β, ν, θ_α, θ_μ) = analytic_parameters(x, ζ, ramp, t)

    Δϵ = β * ν + H * μ * α + Λ * α * ν + x * sign(totalparity)
    return Δϵ
end

"""
    analytic_parameters(x, ζ, ramp, t)

Calculate some useful numbers related to exactly diagonalizing the system
"""
function analytic_parameters(x, ζ, ramp, t)
    Δs = ramp(t)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    ρ = Δ23 / Δ
    η = -ζ^2

    H = η * ρ^2 + x * √(1 - ρ^2)
    Λ = ρ * (x - √(1 - ρ^2) * η)
    θ_μ = -1 / 2 * atan(2 * Λ * H, 1 + Λ^2 - H^2)
    θ_α = atan(H * tan(θ_μ) - Λ)
    ν, μ = sincos(θ_μ)
    β, α = sincos(θ_α)
    return (; H, Λ, μ, α, β, ν, θ_α, θ_μ)
end
