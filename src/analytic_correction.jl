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
function find_zero_energy_from_analytics_midpoint(ζ, ramp, totalparity; kwargs...)
    η = ζ^2
    ϕ = atan(η)
    λ = totalparity * sin(ϕ)
end
function find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity; atol=0.0, rtol=0.0, kwargs...)
    λ = find_zero(λ -> analytic_parameters(totalparity * λ, ζ, ramp, t).ε, initial; atol, rtol, kwargs...)
    return λ
end

# """ 
#     energy_splitting(x, ζ, ramp, t, totalparity)

# Calculate (analytically) the energy splitting between the two lowest energy levels of the system. Works only when all ζs are the same.
# """
# function energy_splitting(λ, ζ, ramp, t, totalparity)
#     (; ηtilde, λtilde, μ, α, β, ν, θ_α, θ_μ) = analytic_parameters(λ, ζ, ramp, t)

#     Δϵ = β * ν + ηtilde * μ * α + λtilde * α * ν - λ * totalparity
#     Δϵ = ηtilde * α - totalparity * λ * μ
#     return Δϵ
# end

"""
    analytic_parameters(x, ζ, ramp, t)

Calculate the energy parameters H and Λ for the system.
Λ (capital λ) and H (capital η) are the generalizations of λ and η for Δ_1 > 0.
In the limit Δ_1 = 0, Λ = λ and H = η.
"""
function analytic_parameters(λ, ζ, ramp, t)
    Δs = ramp(t) ./ (1, sqrt(1 + ζ^4), sqrt(1 + ζ^4)) # divide to normalize the hamiltonian
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    θ = atan(Δ23, Δs[1])

    η = ζ^2
    ηtilde = η * sin(θ)^2 - λ * cos(θ)
    λtilde = λ * sin(θ) + η * cos(θ) * sin(θ)
    θ_μ = -1 / 2 * atan(2 * λtilde * ηtilde, 1 + λtilde^2 - ηtilde^2)
    θ_α = atan(ηtilde * tan(θ_μ) - λtilde)

    ν, μ = sincos(θ_μ)
    β, α = sincos(θ_α)
    Δtilde = (α * μ + ηtilde * β * ν - λtilde * β * μ) * √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    ε = ηtilde * α / μ
    return (; ηtilde, λtilde, μ, α, β, ν, θ_α, θ_μ, Δtilde, ε)
end

function analytic_parameters_midpoint(ζ, ramp, totalparity)
    ## TODO: Implement this function
    #λ = find_zero_energy_from_analytics_midpoint(ζ, ramp, totalparity)
    #analytic_parameters_midpoint(λ, ζ, ramp)
end

@testitem "Zero energy solution" begin
    T = 1e3
    k = 10
    ramp = RampProtocol([0, 0, 0], [1, 1, 1], T, k)
    ζ = 0.5
    for totalparity in (-1, 1)
        @test MajoranaBraiding.find_zero_energy_from_analytics_midpoint(ζ, ramp, totalparity) ≈ find_zero_energy_from_analytics(ζ, ramp, T / 2, 0.0, totalparity)
    end
end

@testitem "Energy splitting" begin
    using StaticArrays, LinearAlgebra
    γ = get_majorana_basis()
    N = length(γ.fermion_basis)
    mtype, vtype = SMatrix{2^(N - 1),2^(N - 1),ComplexF64}, SVector{2^(N - 1)}
    U0 = mtype(I(2^(N - 1)))

    param_dict = Dict(
        :ζ => 0.7, #Majorana overlaps. Number or triplet of numbers
        :ϵs => (0, 0, 0), #Dot energy levels
        :T => 1e4, #Maximum time
        :Δmax => 1 * (rand(3) .+ 0.5), #Largest values of Δs. Number or triplet of numbers
        :Δmin => 1e-10 * (rand(3) .+ 0.5), #Smallest values of Δs. Number or triplet of numbers
        :k => 2e1, #Determines the slope of the ramp
        :steps => 2000, #Number of timesteps for interpolations
        :correction => NoCorrection(), #Different corrections are available. This is the most relevant one for the paper
        :interpolate_corrected_hamiltonian => false, #Creating an interpolated Hamiltonian might speed things up
        :γ => γ, #Majorana basis
        :u0 => U0, #Initial state. Use U0 for the identity matrix.
        :totalparity => 1
    )
    prob = setup_problem(param_dict)
    ts = range(0, 2prob[:T], 1000)
    energy_gaps = map(t -> diff(eigvals(prob[:H](prob[:p], t)))[1], ts)
    energy_gaps2 = map(t -> diff(eigvals(-1im * prob[:op](prob[:u0], prob[:p], t)))[1], ts)
    energy_gaps_analytic = [MajoranaBraiding.energy_splitting(0, prob[:ζ], prob[:ramp], t, prob[:totalparity]) for t in ts]
    plot(energy_gaps)
    plot!(energy_gaps2)
    plot!(2energy_gaps_analytic)

    param_dict = Dict(
        :ζ => 0.7, #Majorana overlaps. Number or triplet of numbers
        :ϵs => (0, 0, 0), #Dot energy levels
        :T => 1e4, #Maximum time
        :Δmax => 1 * (rand(3) .+ 0.5), #Largest values of Δs. Number or triplet of numbers
        :Δmin => 1e-10 * (rand(3) .+ 0.5), #Smallest values of Δs. Number or triplet of numbers
        :k => 2e1, #Determines the slope of the ramp
        :steps => 2000, #Number of timesteps for interpolations
        :correction => InterpolatedExactSimpleCorrection(), #Different corrections are available. This is the most relevant one for the paper
        :interpolate_corrected_hamiltonian => true, #Creating an interpolated Hamiltonian might speed things up
        :γ => γ, #Majorana basis
        :u0 => U0, #Initial state. Use U0 for the identity matrix.
        :totalparity => 1
    )

    energy_gaps = map(t -> diff(eigvals(prob[:H](prob[:p], t)))[1], range(0, 2prob[:T], 1000))
    @test all(abs.(energy_gaps) .< 1e-10)

end