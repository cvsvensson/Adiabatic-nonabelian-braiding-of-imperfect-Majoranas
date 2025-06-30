struct InterpolatedExactSimpleCorrection <: AbstractCorrection end

function setup_correction(::InterpolatedExactSimpleCorrection, d::Dict)
    ζ = d[:ζ]
    ramp = d[:ramp]
    ts = d[:ts]
    return analytical_exact_simple_correction(ζ, ramp, ts, d[:totalparity])
end
function analytical_exact_simple_correction(ζ, ramp, ts, totalparity; opt_kwargs...)
    results = Float64[]
    for t in ts
        # Find roots of the energy split function
        initial = 0.0#length(results) > 0 ? results[end] : 0.0
        result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity; opt_kwargs...)
        push!(results, result)
    end
    return SimpleCorrection(linear_interpolation(ts, results, extrapolation_bc=Periodic()))
end

find_zero_energy_from_analytics(ζ::Tuple, ramp, t, initial, totalparity; kwargs...) = find_zero_energy_from_analytics(effective_ζ_by_η(ζ), ramp, t, initial, totalparity; kwargs...)
function find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity; atol=1e-15, rtol=0.0, kwargs...)
    λ = try
        find_zero(λ -> analytic_parameters(λ, ζ, ramp, t).ε + totalparity * λ, initial; atol, rtol, kwargs...)
    catch
        find_zero(λ -> analytic_parameters(λ, ζ, ramp, t).ε + totalparity * λ, initial; atol, rtol, verbose=true, kwargs...)
    end
    return λ
end


function analytic_energy_spectrum(λ, ζ, ramp, t, totalparity)
    (; Δ, ε, Δtilde) = analytic_parameters(λ, ζ, ramp, t)
    es = [0, ε + λ, ε + Δtilde, Δtilde + λ] .- (totalparity == -1) * ε
    2Δ * sort(es .- sum(es) / 4)
end

# effective_ζ(ζ::Tuple) = (ζ[2] + ζ[3]) / 2 # bad because ζ[1] also effects the result
# effective_ζ(ζ::Tuple) = (ζ[1] +ζ[2] +ζ[3])/3 #very bad
# effective_ζ(ζ::Tuple) = (ζ[1]ζ[2]ζ[3])^(1/3) #not good
# effective_ζ(ζ::Tuple) = (ζ[1] + sqrt(ζ[2]ζ[3])) / 2 # meh
# effective_ζ(ζ::Tuple) = sqrt(ζ[1] * (ζ[2] + ζ[3]) / 2)  #nice
# effective_ζ(ζ::Tuple) = (sqrt(ζ[1] * ζ[2]) + sqrt(ζ[1] * ζ[3])) / 2 #best so far
function effective_ζ_by_η(ζ::Tuple)
    ηs = map(x -> x^2, ζ)
    # return (sqrt(ηs[1]) * (sqrt(ηs[2]) + sqrt(ηs[3])) / 2) |> sqrt
    # return (sqrt(ηs[1]) * sqrt((ηs[2] + ηs[3]) / 2)) |> sqrt
    # return (sqrt(ηs[1]) * (ηs[2] * ηs[3])^(1 / 4)) |> sqrt
    return sqrt(ηs[1] * sqrt(ηs[2] * ηs[3])) |> sqrt # awesome
end

analytic_parameters(λ, ζ::Tuple, ramp, t) = analytic_parameters(λ, effective_ζ_by_η(ζ), ramp, t)
function analytic_parameters(λ, ζ, ramp, t)
    Δs = ramp(t) ./ (1, (1 + ζ^2), (1 + ζ^2)) # divide to normalize the hamiltonian
    Δ = sqrt(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    θ = atan(Δ23, Δs[1])
    ϕ = atan(Δs[3], Δs[2])
    η = ζ^2
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

# analytic_parameters_midpoint(ζ::Tuple, totalparity) = analytic_parameters_midpoint(effective_ζ_by_η(ζ), totalparity)
# function analytic_parameters_midpoint(ζ, totalparity)
#     η = ζ^2
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

analytical_components_middle_of_protocol(d::Dict) = analytical_components_middle_of_protocol(d[:ζ], d[:totalparity])
analytical_components_middle_of_protocol(ζ::Tuple, totalparity) = analytical_components_middle_of_protocol(effective_ζ_by_η(ζ), totalparity)
function analytical_components_middle_of_protocol(ζ, totalparity)
    η = ζ^2
    λ = -totalparity * η / sqrt(1 + η^2)
    θ_μ = -1 / 2 * atan(2 * λ * η / (1 + λ^2 - η^2))
    ν, μ = sincos(θ_μ)
    θ_α = -1 * atan(-η * tan(θ_μ) + λ)
    β, α = sincos(θ_α)
    return (; μ, α, β, ν, θ_α, θ_μ, λ)
end


@testitem "Energy splitting" begin
    using StaticArrays, LinearAlgebra
    γ = get_majorana_basis()
    N = length(γ.fermion_basis)
    mtype, vtype = SMatrix{2^(N - 1),2^(N - 1),ComplexF64}, SVector{2^(N - 1)}

    T = 10
    k = 1
    ramp = RampProtocol(0, 1, T, k)
    λ = 0.2
    corr = SimpleCorrection(λ)
    totalparity = -1
    P = parity_operators(γ, totalparity, mtype)
    ζ = 0.5
    p = (ramp, (0, 0, 0), ζ, corr, P)

    spectrum = stack(map(t -> eigvals(-totalparity * ham_with_corrections(p, t)), ts))' # Why the totalparity here?
    analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, ζ, ramp, t, totalparity), ts))'
    @test spectrum ≈ analytic_spectrum

    param_dict = Dict(
        :ζ => 0.7, #Majorana overlaps. Number or triplet of numbers
        :T => 1e4, #Maximum time
        :Δmax => 1 * (rand(3) .+ 0.5), #Largest values of Δs. Number or triplet of numbers
        :Δmin => 1e-10 * (rand(3) .+ 0.5), #Smallest values of Δs. Number or triplet of numbers
        :k => 2e1, #Determines the slope of the ramp
        :steps => 2000, #Number of timesteps for interpolations
        :correction => NoCorrection(), #Different corrections are available. This is the most relevant one for the paper
        :interpolate_corrected_hamiltonian => false, #Creating an interpolated Hamiltonian might speed things up
        :γ => γ, #Majorana basis
        :initial => I, #Initial state. Use U0 for the identity matrix.
        :totalparity => 1
    )
    prob = setup_problem(param_dict)
    ts = range(0, 2prob[:T], 10)
    λ = 0
    spectrum = stack(map(t -> eigvals(prob[:H](prob[:p], t)), ts))'
    analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, prob[:ζ], prob[:ramp], t, prob[:totalparity]), ts))'
    @test norm(spectrum - analytic_spectrum) < 1e-10

    param_dict[:totalparity] = -1
    prob = setup_problem(param_dict)
    spectrum = stack(map(t -> eigvals(prob[:H](prob[:p], t)), ts))'
    analytic_spectrum = stack(map(t -> MajoranaBraiding.analytic_energy_spectrum(λ, prob[:ζ], prob[:ramp], t, prob[:totalparity]), ts))'
    @test spectrum ≈ analytic_spectrum


    energy_gaps = map(t -> diff(eigvals(prob[:H](prob[:p], t)))[1], ts)
    energy_gaps2 = map(t -> diff(eigvals(-1im * prob[:op](prob[:u0], prob[:p], t)))[1], ts)
    energy_gaps_analytic = [2MajoranaBraiding.analytic_parameters(0, prob[:ζ], prob[:ramp], t).ε for t in ts]
    plot(energy_gaps)
    plot!(energy_gaps2)
    plot!(energy_gaps_analytic)

    param_dict = Dict(
        :ζ => 0.7, #Majorana overlaps. Number or triplet of numbers
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