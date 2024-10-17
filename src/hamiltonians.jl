
ham_with_corrections(p, t, α=1) = _ham_with_corrections(p..., t, α)

function _ham_with_corrections(ramp, ϵs, ζs, correction, P, t, α=1)
    Δs = ramp(t)
    Ham = (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
           ϵs[1] * P[0, 1] + ϵs[2] * P[2, 4] + ϵs[3] * P[3, 5] +
           _error_ham(Δs, ζs, P))
    Ham += correction(t, Δs, ζs, P, Ham)
    return Ham * α
end
_error_ham(Δs, ζ::Number, P) = _error_ham(Δs, (ζ, ζ, ζ), P)
_error_ham(Δs, ζs, P) = +Δs[2] * ζs[1] * ζs[2] * P[1, 4] + Δs[3] * ζs[1] * ζs[3] * P[1, 5]
_error_ham(ramp, t, ζ::Number, P) = _error_ham(ramp, t, (ζ, ζ, ζ), P)
function _error_ham(ramp, t, ζs, P)
    Δs = ramp(t)
    +Δs[2] * ζs[1] * ζs[2] * P[1, 4] + Δs[3] * ζs[1] * ζs[3] * P[1, 5]
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
setup_correction(corr::SimpleCorrection, ::Dict) = corr

struct IndependentSimpleCorrections{T} <: AbstractCorrection
    scaling::T
end
function IndependentSimpleCorrections(scaling1, scaling2)
    newscaling1 = _process_constant_scaling(scaling1)
    newscaling2 = _process_constant_scaling(scaling2)
    IndependentSimpleCorrections(t -> (newscaling1(t), newscaling2(t)))
end
IndependentSimpleCorrections(scalings::Vector{<:Number}) = length(scalings) == 2 ? IndependentSimpleCorrections(scalings...) : error("scalings must be a vector of length 2")
setup_correction(corr::IndependentSimpleCorrections, ::Dict) = corr

_process_constant_scaling(scaling::Number) = t -> scaling
_process_constant_scaling(scaling) = scaling

function (corr::IndependentSimpleCorrections)(t, Δs, ζs, P, ham)
    Δ = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    scaling = corr.scaling(t)
    scaling[1] * Δ * P[2, 4] + scaling[2] * Δ * P[3, 5]
end
struct CorrectionSum
    corrections::Vector{<:AbstractCorrection}
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
setup_correction(corr::CorrectionSum, d::Dict) = CorrectionSum(setup_correction.(corr.corrections, d))


struct EigenEnergyCorrection{T} <: AbstractCorrection
    scaling::T
    function EigenEnergyCorrection(scaling)
        newscaling = _process_constant_scaling(scaling)
        new{typeof(newscaling)}(newscaling)
    end
end
EigenEnergyCorrection() = EigenEnergyCorrection(t -> true)
(corr::EigenEnergyCorrection)(t, Δs, ζs, P, ham) = iszero(corr.scaling(t)) ? zero(ham) : (corr.scaling(t) * full_energy_correction_term(ham))
setup_correction(corr::EigenEnergyCorrection, ::Dict) = corr

function full_energy_correction_term(ham)
    vals, vecs = eigen(Hermitian(ham))
    δE = (vals[2] - vals[1]) / 2
    δE * (vecs[:, 1] * vecs[:, 1]' - vecs[:, 2] * vecs[:, 2]')
end

struct WeakEnergyCorrection{B,T} <: AbstractCorrection
    basis::B
    scaling::T
    function WeakEnergyCorrection(basis, scaling)
        newscaling = _process_constant_scaling(scaling)
        new{typeof(basis),typeof(newscaling)}(basis, newscaling)
    end
end
setup_correction(corr::WeakEnergyCorrection, ::Dict) = corr
WeakEnergyCorrection(basis) = WeakEnergyCorrection(basis, t -> true)
(corr::WeakEnergyCorrection)(t, Δs, ζs, P, ham) = iszero(corr.scaling(t)) ? zero(ham) : (corr.scaling(t) * weak_energy_correction_term(ham, corr.basis))

function weak_energy_correction_term(ham, basis; alg=Majoranas.WM_BACKSLASH())
    vals, vecs = eigen(Hermitian(ham))
    δE = (vals[2] - vals[1]) / 2
    # push the lowest energy states δE closer together
    weak_ham_prob = WeakMajoranaProblem(basis, vecs, nothing, [nothing, nothing, nothing, δE])
    sol = solve(weak_ham_prob, alg)
    return Majoranas.coeffs_to_matrix(basis, sol)
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
