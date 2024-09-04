
ham_with_corrections(p, t, α=1) = _ham_with_corrections(p..., t, α)

function _ham_with_corrections(ramp, ϵs, ζs, correction, P, t, α=1)
    Δs = ramp(t)
    Ham = (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
           ϵs[1] * P[0, 1] + ϵs[2] * P[2, 4] + ϵs[3] * P[3, 5] +
           -Δs[2] * ζs[1] * ζs[2] * P[1, 4] - Δs[3] * ζs[1] * ζs[3] * P[1, 5])
    Ham += correction(t, Δs, ζs, P, Ham)
    return Ham * α
end


abstract type AbstractCorrection end
(corr::AbstractCorrection)(t, Δs, ζs, P, ham) = error("(corr::C)(t, Δs, ζs, P, ham) not implemented for C=$(typeof(corr))")
struct NoCorrection <: AbstractCorrection end
(corr::NoCorrection)(t, Δs, ζs, P, ham) = 0I
struct SimpleCorrection{T} <: AbstractCorrection
    scaling::T
    function SimpleCorrection(scaling)
        newscaling = _process_constant_scaling(scaling)
        new{typeof(newscaling)}(newscaling)
    end
end
SimpleCorrection() = SimpleCorrection(true)
SimpleCorrection(scaling::Number) = SimpleCorrection(t -> scaling)
(corr::SimpleCorrection)(t, Δs, ζs, P, ham) = corr.scaling(t) * √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2) * (P[2, 4] + P[3, 5])

struct IndependentSimpleCorrections{T} <: AbstractCorrection
    scaling::T
end
function IndependentSimpleCorrections(scaling1, scaling2)
    newscaling1 = _process_constant_scaling(scaling1)
    newscaling2 = _process_constant_scaling(scaling2)
    IndependentSimpleCorrections(t -> (newscaling1(t), newscaling2(t)))
end
IndependentSimpleCorrections(scalings::Vector{<:Number}) = length(scalings) == 2 ? IndependentSimpleCorrections(scalings...) : error("scalings must be a vector of length 2")
_process_constant_scaling(scaling::Number) = t -> scaling
_process_constant_scaling(scaling) = scaling

function (corr::IndependentSimpleCorrections)(t, Δs, ζs, P, ham)
    Δ = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    scaling = corr.scaling(t)
    scaling[1] * Δ * P[2, 4] + scaling[2] * Δ * P[3, 5]
end
struct CorrectionSum
    corrections::Vector{<:AbstractCorrection}
    function CorrectionSum(corrections)
        new(sort(corrections))
    end
end
Base.:+(corr1::AbstractCorrection, corr2::AbstractCorrection) = CorrectionSum([corr1, corr2])
Base.:+(corr::CorrectionSum, corr2::AbstractCorrection) = CorrectionSum([corr.corrections..., corr2])
Base.:+(corr1::AbstractCorrection, corr::CorrectionSum) = CorrectionSum([corr1, corr.corrections...])

function (corr::CorrectionSum)(args...)
    ham0 = args[end]
    pre_args = args[1:end-1]
    function f((old_corr, old_ham), _corr)
        new_corr = _corr(pre_args..., old_ham)
        (old_corr + new_corr, old_ham + new_corr)
    end
    foldl(f, corr.corrections, init=(0I, ham0))[1]
end


struct EigenEnergyCorrection{T} <: AbstractCorrection
    scaling::T
    function EigenEnergyCorrection(scaling)
        newscaling = _process_constant_scaling(scaling)
        new{typeof(newscaling)}(newscaling)
    end
end
EigenEnergyCorrection() = EigenEnergyCorrection(t -> true)
(corr::EigenEnergyCorrection)(t, Δs, ζs, P, ham) = iszero(corr.scaling(t)) ? zero(ham) : (corr.scaling(t) * full_energy_correction_term(ham, P))
Base.isless(::EigenEnergyCorrection, ::AbstractCorrection) = false
Base.isless(::AbstractCorrection, ::EigenEnergyCorrection) = true

function full_energy_correction_term(ham, P)
    vals, vecs = eigen(Hermitian(ham))
    δE = (vals[2] - vals[1]) / 2
    weak_ham_prob = WeakMajoranaProblem(P, vecs, 0, δE, nothing) # push the lowest energy states δE closer together
    sol = only(solve(weak_ham_prob, Majoranas.WM_BACKSLASH_SPARSE()))
    #=δv = (vecs[:, 1] * vecs[:, 1]' - vecs[:, 2] * vecs[:, 2]') + =#
    #=     (vecs[:, 3] * vecs[:, 3]' - vecs[:, 4] * vecs[:, 4]')=#
    return Majoranas.coeffs_to_matrix(P, sol)
end

function optimized_simple_correction(H, (ramp, ϵs, ζs, P), ts; alg=BFGS())
    results = Float64[]
    function cost_function(x, t)
        vals = eigvals(H((ramp, ϵs, ζs, SimpleCorrection(x), P), t))
        return vals[2] - vals[1]
    end
    for t in ts
        f(x) = cost_function(only(x), t)
        initial = length(results) > 0 ? results[end] : 1.0
        result = optimize(f, [initial], alg, Optim.Options(time_limit=1 / length(ts)))
        push!(results, only(result.minimizer))
    end
    return SimpleCorrection(linear_interpolation(ts, results))
end

function optimized_independent_simple_correction(H, (ramp, ϵs, ζs, P), ts; alg=BFGS())
    results = Vector{Float64}[]
    # define a cost function that x as a vector instead of a scalar
    function cost_function(x::Vector, t)
        vals = eigvals(H((ramp, ϵs, ζs, IndependentSimpleCorrections(x), P), t))
        return vals[2] - vals[1]
    end
    abs_err = 1e-10
    rel_err = 1e-10
    for t in ts
        initial = length(results) > 0 ? results[end] : [0.0, 0.0]
        result = optimize(x -> cost_function(x, t), initial, alg,
            Optim.Options(time_limit=1 / length(ts)))#, Optim.Options(g_tol=abs_err, x_tol=rel_err))
        push!(results, result.minimizer)
    end
    return IndependentSimpleCorrections(linear_interpolation(ts, results))
end

function analytical_exact_simple_correction(ζ, ramp, ts)
    results = Float64[]
    η = ζ^2
    for t in ts
        # Find roots of the energy split function
        initial = length(results) > 0 ? results[end] : 0.0
        result = find_zero(x -> energy_split(x, η, ramp, t), initial)
        push!(results, result)
    end
    return SimpleCorrection(linear_interpolation(ts, results))
end

function energy_split(x, η, ramp, t)
    Δs = ramp(t)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    ρ = Δ23 / Δ

    Η = η * ρ^2 + x * √(1 - ρ^2)
    Λ = ρ * x - ρ * √(1 - ρ^2) * η

    χ = 4 * Λ^2 * Η^2 / ((1 + Λ^2 - Η^2)^2 + 4 * Λ^2 * Η^2)
    μ = 1 / √(2) * √(1 + √(1 - χ))
    ν = 1 / √(2) * √(1 - √(1 - χ))
    α = (Η * μ + Λ * ν) / √((Η * μ + Λ * ν)^2 + ν^2)
    β = ν / √((Η * μ + Λ * ν)^2 + ν^2)
    vals = β * ν + Η * μ * α + Λ * α * ν + x
    return vals
end

function groundstate_components(x, η, ramp, t)
    Δs = ramp(t)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    ρ = Δ23/ Δ

    Η = η * ρ^2 + x * √( 1 - ρ^2 )
    Λ = ρ * x - ρ * √( 1- ρ^2 ) * η

    χ = 4*Λ^2 * Η^2 / ( (1+ Λ^2 - Η^2)^2 + 4*Λ^2 * Η^2 )
    μ = 1/ √(2) * √(1 + √(1 - χ))
    ν = 1/ √(2) * √(1 - √(1 - χ))
    α = (Η * μ + Λ * ν)/ √( (Η * μ + Λ * ν)^2 + ν^2 )
    β = ν/ √( (Η * μ + Λ * ν)^2 + ν^2 )

    return μ, α, β, ν
end

function single_braid_gate_improved(P, ζ, ramp, T)
    t = T/2
    η = ζ^2
    initial = 0.0
    result = find_zero(x -> MajoranaBraiding.energy_split(x, η, ramp, t), initial, xtol=1e-6)
    μ, α, β, ν = MajoranaBraiding.groundstate_components(result, η, ramp, t)
    θ = atan(μ, ν) / 2
    ϕ = atan(β, α) / 2
    unit = Diagonal([1, 1, 1, 1])
    return ( cos(θ) * unit + sin(θ) *  (1im) * P[2, 3] ) * (cos(ϕ) * unit + sin(ϕ) * (1im) * P[4, 5])
end
