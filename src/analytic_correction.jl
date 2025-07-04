struct InterpolatedExactSimpleCorrection <: AbstractCorrection end

function setup_correction(::InterpolatedExactSimpleCorrection, d::Dict)
    return analytical_exact_simple_correction(d[:η], d[:k], d[:ts], d[:totalparity])
end
function analytical_exact_simple_correction(η, k, ts, totalparity; opt_kwargs...)
    results = Float64[]
    for t in ts
        # Find roots of the energy split function
        initial = 0.0#length(results) > 0 ? results[end] : 0.0
        result = find_zero_energy_from_analytics(η, k, t, initial, totalparity; opt_kwargs...)
        push!(results, result)
    end
    return SimpleCorrection(linear_interpolation(ts, results, extrapolation_bc=Periodic()))
end

find_zero_energy_from_analytics(η::Tuple, k, t, initial, totalparity; kwargs...) = find_zero_energy_from_analytics(effective_η(η), k, t, initial, totalparity; kwargs...)
function find_zero_energy_from_analytics(η, k, t, initial, totalparity; atol=1e-15, rtol=0.0, kwargs...)
    λ = try
        find_zero(λ -> analytic_parameters(λ, η, k, t).ε + totalparity * λ, initial; atol, rtol, kwargs...)
    catch
        find_zero(λ -> analytic_parameters(λ, η, k, t).ε + totalparity * λ, initial; atol, rtol, verbose=true, kwargs...)
    end
    return λ
end


function analytic_energy_spectrum(λ, η, k, t, totalparity)
    (; Δ, ε, Δtilde) = analytic_parameters(λ, η, k, t)
    es = [0, ε + λ, ε + Δtilde, Δtilde + λ] .- (totalparity == -1) * ε
    2Δ * sort(es .- sum(es) / 4)
end

function effective_η(ηs::Tuple)
    return sqrt(ηs[1] * sqrt(ηs[2] * ηs[3])) #|> sqrt # awesome
end

analytic_parameters(λ, η::Tuple, k, t) = analytic_parameters(λ, effective_η(η), k, t)
function analytic_parameters(λ, η, k, t)
    ρs = get_rhos(k, t) #./ (1, (1 + η), (1 + η)) # divide to normalize the hamiltonian
    Δ = 1#sqrt(ρs[1]^2 + ρs[2]^2 + ρs[3]^2)
    ρ23 = √(ρs[2]^2 + ρs[3]^2)
    θ = atan(ρ23, ρs[1])
    ϕ = atan(ρs[3], ρs[2])
    λtilde = λ * sin(θ) + η * cos(θ) * sin(θ)
    ηtilde = η * sin(θ)^2 - λ * cos(θ)
    θ_μ = -1 / 2 * atan(2 * λtilde * ηtilde, 1 + λtilde^2 - ηtilde^2)
    θ_α = atan(ηtilde * tan(θ_μ) - λtilde)

    ν, μ = sincos(θ_μ)
    β, α = sincos(θ_α)
    Δtilde = (α * μ + ηtilde * β * ν - λtilde * β * μ)

    ε = β * ν + λtilde * α * ν + ηtilde * α * μ # = ηtilde * α / μ. The sign of the last term differs from the SI.

    return (; ηtilde, λtilde, μ, α, β, ν, θ_α, θ_μ, Δtilde, ε, Δ, θ, ϕ, η, λ)
end

# analytic_parameters_midpoint(η::Tuple, totalparity) = analytic_parameters_midpoint(effective_η_by_η(η), totalparity)
# function analytic_parameters_midpoint(η, totalparity)
#     λ = -totalparity * η / sqrt(1 + η^2)
#     # α = cos(atan(-η * tan(θ) + λ))
#     # α = cos(atan(-η * ν / sqrt(1 - ν^2) + λ))
#     # θ = -1 / 2 * atan(2λ * η, 1 + λ^2 - η^2)
#     # α = 1 / sqrt(1 + (λ - η * tan(θ))^2)
#     # ν = sin(θ)
#     ν = -sin(1 / 2 * atan(2 * λ * η, 1 + λ^2 - η^2))
#     α = cos(atan(-η * ν / sqrt(1 - ν^2) + λ))
#     θ_μ = asin(v)
#     (; ν, α, η, λ)
# end

analytical_components_middle_of_protocol(d::Dict) = analytical_components_middle_of_protocol(d[:η], d[:totalparity])
analytical_components_middle_of_protocol(η::Tuple, totalparity) = analytical_components_middle_of_protocol(effective_η(η), totalparity)
function analytical_components_middle_of_protocol(η, totalparity)
    λ = -totalparity * η / sqrt(1 + η^2)
    θ_μ = -1 / 2 * atan(2 * λ * η / (1 + λ^2 - η^2))
    ν, μ = sincos(θ_μ)
    θ_α = -1 * atan(-η * tan(θ_μ) + λ)
    β, α = sincos(θ_α)
    return (; μ, α, β, ν, θ_α, θ_μ, λ)
end


@testitem "Energy splitting" begin
    using StaticArrays, LinearAlgebra
    N = 3
    mtype, vtype = SMatrix{2^(N - 1),2^(N - 1),ComplexF64}, SVector{2^(N - 1)}

    T = 10
    k = 1
    λ = 0.2
    corr = SimpleCorrection(λ)
    totalparity = -1
    P = parity_operators(totalparity, mtype)
    η = 0.5
    p = (k, T, (0, 0, 0), η, corr, P)

    spectrum = stack(map(t -> eigvals(-totalparity * ham_with_corrections(p, t)), ts))' # Why the totalparity here?
    analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, η, k, t / T, totalparity), ts))'
    @test spectrum ≈ analytic_spectrum

    param_dict = Dict(
        :η => 0.7, #Majorana overlaps. Number or triplet of numbers
        :T => 1e4, #Maximum time
        :ρmax => 1 * (rand(3) .+ 0.5), #Largest values of ρs. Number or triplet of numbers
        :ρmin => 1e-10 * (rand(3) .+ 0.5), #Smallest values of ρs. Number or triplet of numbers
        :k => 2e1, #Determines the slope of the ramp
        :steps => 2000, #Number of timesteps for interpolations
        :correction => NoCorrection(), #Different corrections are available. This is the most relevant one for the paper
        :interpolate_corrected_hamiltonian => false, #Creating an interpolated Hamiltonian might speed things up
        :γ => γ, #Majorana basis
        :initial => I, #Initial state. Use U0 for the identity matrix.
        :totalparity => 1
    )
    prob = setup_problem(param_dict)
    ts = range(0, 2, 10)
    λ = 0
    spectrum = stack(map(t -> eigvals(prob[:H](prob[:p], t)), ts))'
    analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, prob[:η], prob[:k], t, prob[:totalparity]), ts))'
    @test norm(spectrum - analytic_spectrum) < 1e-10

    param_dict[:totalparity] = -1
    prob = setup_problem(param_dict)
    spectrum = stack(map(t -> eigvals(prob[:H](prob[:p], t)), ts))'
    analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, prob[:η], prob[:k], t, prob[:totalparity]), ts))'
    @test spectrum ≈ analytic_spectrum


    energy_gaps = map(t -> diff(eigvals(prob[:H](prob[:p], t)))[1], ts)
    energy_gaps2 = map(t -> diff(eigvals(-1im * prob[:op](prob[:u0], prob[:p], t)))[1], ts)
    energy_gaps_analytic = [2MajoranaBraiding.analytic_parameters(0, prob[:ζ], prob[:k], t).ε for t in ts]
    plot(energy_gaps)
    plot!(energy_gaps2)
    plot!(energy_gaps_analytic)

    param_dict = Dict(
        :η => 0.7, #Majorana overlaps. Number or triplet of numbers
        :T => 1e4, #Maximum time
        :ρmax => 1 * (rand(3) .+ 0.5), #Largest values of ρs. Number or triplet of numbers
        :ρmin => 1e-10 * (rand(3) .+ 0.5), #Smallest values of ρs. Number or triplet of numbers
        :k => 2e1, #Determines the slope of the ramp
        :steps => 2000, #Number of timesteps for interpolations
        :correction => InterpolatedExactSimpleCorrection(), #Different corrections are available. This is the most relevant one for the paper
        :interpolate_corrected_hamiltonian => true, #Creating an interpolated Hamiltonian might speed things up
        :γ => γ, #Majorana basis
        :u0 => U0, #Initial state. Use U0 for the identity matrix.
        :totalparity => 1
    )

    energy_gaps = map(t -> diff(eigvals(prob[:H](prob[:p], t)))[1], range(0, 2, 1000))
    @test all(abs.(energy_gaps) .< 1e-10)

end