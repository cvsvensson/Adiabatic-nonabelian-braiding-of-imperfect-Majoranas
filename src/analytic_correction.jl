struct InterpolatedExactSimpleCorrection <: AbstractCorrection end

function setup_correction(corr::InterpolatedExactSimpleCorrection, d::Dict)
    ζ = d[:ζ]
    ramp = d[:ramp]
    ts = d[:ts]
    return analytical_exact_simple_correction(ζ, ramp, ts, d[:totalparity])
end
function analytical_exact_simple_correction(ζ, ramp, ts, totalparity; opt_kwargs...)
    results = Float64[]
    for t in ts
        # Find roots of the energy split function
        initial = length(results) > 0 ? results[end] : 0.0
        result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity; opt_kwargs...)
        push!(results, result)
    end
    return SimpleCorrection(linear_interpolation(ts, results, extrapolation_bc=Periodic()))
end

function find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity; kwargs...)
    result = find_zero(x -> energy_splitting(x, ζ, ramp, t, totalparity), initial; kwargs...)
    return result
end

""" 
    energy_splitting(x, ζ, ramp, t, totalparity=1)

Calculate (analytically) the energy splitting between the two lowest energy levels of the system. Works only when all ζs are the same?
Yes it works only when all ζs are the same.
"""
function energy_splitting(x, ζ, ramp, t, totalparity)
    (; η_gen, λ_gen, μ, α, β, ν, θ_α, θ_μ) = analytic_parameters(x, ζ, ramp, t)

    Δϵ = β * ν + η_gen * μ * α + λ_gen * α * ν - x * totalparity
    Δϵ = η_gen * α - totalparity * x * μ
    return Δϵ
end

"""
    analytic_parameters(x, ζ, ramp, t)

Calculate the energy parameters H and Λ for the system.
Λ (capital λ) and H (capital η) are the generalizations of λ and η for Δ_1 > 0.
In the limit Δ_1 = 0, Λ = λ and H = η.
"""
function analytic_parameters(x, ζ, ramp, t)
    Δs = ramp(t) ./ (1, sqrt(1 + ζ^4), sqrt(1 + ζ^4)) # divide to normalize the hamiltonian
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    θ_spherical = atan(Δ23, Δs[1])

    η = ζ^2
    η_gen = η * sin(θ_spherical)^2 - x * cos(θ_spherical)
    λ_gen = sin(θ_spherical) * x + cos(θ_spherical) * sin(θ_spherical) * η
    θ_μ = -1 / 2 * atan(2 * λ_gen * η_gen, 1 + λ_gen^2 - η_gen^2)
    θ_α = atan(η_gen * tan(θ_μ) - λ_gen)

    ν, μ = sincos(θ_μ)
    β, α = sincos(θ_α)
    return (; η_gen, λ_gen, μ, α, β, ν, θ_α, θ_μ)
end
