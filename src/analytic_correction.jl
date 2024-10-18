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
function analytical_exact_simple_correction(ζ, ramp, ts, totalparity=1)
    results = Float64[]
    for t in ts
        # Find roots of the energy split function
        initial = length(results) > 0 ? results[end] : 0.0
        result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity)
        push!(results, result)
    end
    return SimpleCorrection(linear_interpolation(ts, results, extrapolation_bc=Periodic()))
end


function find_zero_energy_from_analytics(ζ, ramp, t, initial=0.0, totalparity=1)
    result = find_zero(x -> energy_splitting(x, ζ, ramp, t, totalparity), initial)
    return result
end

""" 
    energy_splitting(x, ζ, ramp, t, totalparity=1)

Calculate (analytically) the energy splitting between the two lowest energy levels of the system. Works only when all ζs are the same?
"""
function energy_splitting(x, ζ, ramp, t, totalparity=1)
    Η, Λ = energy_parameters(x, ζ, ramp, t)
    μ, α, β, ν = groundstate_components(x, ζ, ramp, t)

    Δϵ = β * ν + Η * μ * α + Λ * α * ν + x * sign(totalparity)
    return Δϵ
end

""" 
    energy_parameters(x, ζ, ramp, t)

Calculate the energy parameters Η and Λ for the system. What do they mean?
"""
function energy_parameters(x, ζ, ramp, t)
    Δs = ramp(t)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    ρ = Δ23 / Δ
    η = -ζ^2

    Η = η * ρ^2 + x * √(1 - ρ^2)
    Λ = ρ * x - ρ * √(1 - ρ^2) * η
    return Η, Λ
end

"""
    groundstate_components(x, ζ, ramp, t)

Calculate the components ???
"""
function groundstate_components(x, ζ, ramp, t)
    Η, Λ = energy_parameters(x, ζ, ramp, t)

    θ_μ = -1 / 2 * atan(2 * Λ * Η, 1 + Λ^2 - Η^2)
    μ = cos(θ_μ)
    ν = sin(θ_μ)

    θ_α = atan(Η * tan(θ_μ) - Λ)
    α = cos(θ_α)
    β = sin(θ_α)
    return μ, α, β, ν
end
