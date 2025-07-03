
ham_with_corrections(p, t, α=1) = _ham_with_corrections(p..., t, α)

_ham_with_corrections(ramp, η::Number, correction, P, t, α=1) = _ham_with_corrections(ramp, (η, η, η), correction, P, t, α)
function _ham_with_corrections(ramp, ηs, correction, P, t, α=1)
    ρs = ramp(t) #./ (1, sqrt(1 + ηs[1]) * sqrt(1 + ηs[2]), sqrt(1 + ηs[1]) * sqrt(1 + ηs[3])) # divide to normalize the hamiltonian
    Ham = (ρs[1] * P[:M, :M̃] +
           ρs[2] * P[:M, :L] +
           ρs[3] * P[:M, :R] +
           _error_ham(ρs, ηs, P))
    Ham += correction(t, ρs, ηs, P, Ham)
    return Ham * α
end
_error_ham(ρs, η::Number, P) = _error_ham(ρs, (η, η, η), P)
_error_ham(ρs, ηs, P) = ρs[2] * sqrt(ηs[1] * ηs[2]) * P[:M̃, :L̃] + ρs[3] * sqrt(ηs[1] * ηs[3]) * P[:M̃, :R̃]


abstract type AbstractCorrection end
(corr::AbstractCorrection)(t, ρs, ηs, P, ham) = error("(corr::C)(t, ρs, ηs, P, ham) not implemented for C=$(typeof(corr))")
struct NoCorrection <: AbstractCorrection end
(corr::NoCorrection)(t, ρs, ηs, P, ham) = 0I
struct SimpleCorrection{T} <: AbstractCorrection
    scaling::T
    function SimpleCorrection(scaling)
        newscaling = _process_constant_scaling(scaling)
        new{typeof(newscaling)}(newscaling)
    end
end
setup_correction(::NoCorrection, ::Dict) = NoCorrection()

SimpleCorrection() = SimpleCorrection(true)
SimpleCorrection(scaling::Number) = SimpleCorrection(t -> scaling)
(corr::SimpleCorrection)(t, ρs, ηs, P, ham) = corr.scaling(t) * (P[:L, :L̃] + P[:R, :R̃]) #* √(ρs[1]^2 + ρs[2]^2 + ρs[3]^2)
setup_correction(corr::SimpleCorrection, ::Dict) = corr

struct IndependentSimpleCorrection{T} <: AbstractCorrection
    scaling::T
end
function IndependentSimpleCorrection(scaling1, scaling2)
    newscaling1 = _process_constant_scaling(scaling1)
    newscaling2 = _process_constant_scaling(scaling2)
    IndependentSimpleCorrection(t -> (newscaling1(t), newscaling2(t)))
end
IndependentSimpleCorrection(scalings::Vector{<:Number}) = length(scalings) == 2 ? IndependentSimpleCorrection(scalings...) : error("scalings must be a vector of length 2")
setup_correction(corr::IndependentSimpleCorrection, ::Dict) = corr

_process_constant_scaling(scaling::Number) = t -> scaling
_process_constant_scaling(scaling) = scaling

function (corr::IndependentSimpleCorrection)(t, Δs, ηs, P, ham)
    Δ = 1#√(ρs[1]^2 + ρs[2]^2 + ρs[3]^2)
    scaling = corr.scaling(t)
    scaling[1] * Δ * P[:L, :L̃] + scaling[2] * Δ * P[:R, :R̃]
end

struct OptimizedSimpleCorrection <: AbstractCorrection end

function setup_correction(::OptimizedSimpleCorrection, d::Dict)
    return optimized_simple_correction(d[:ramp], d[:η], d[:P], d[:ts])
end
function optimized_simple_correction(ramp, ηs, P, ts; alg=BFGS())
    H = ham_with_corrections
    results = Float64[]
    function cost_function(x, t)
        vals = eigvals(H((ramp, ηs, SimpleCorrection(x), P), t))
        return vals[2] - vals[1]
    end
    for t in ts
        f(x) = cost_function(only(x), t)
        initial = length(results) > 0 ? results[end] : 0.0
        result = optimize(f, [initial], alg, Optim.Options(time_limit=1 / length(ts)))
        # println("result = ", result)
        push!(results, only(result.minimizer))
    end
    return SimpleCorrection(linear_interpolation(ts, results))
end

struct OptimizedIndependentSimpleCorrection <: AbstractCorrection
    maxtime::Float64
    penalty_factor::Float64
end

function setup_correction(corr::OptimizedIndependentSimpleCorrection, d::Dict)
    return optimized_independent_simple_correction(d[:ramp], d[:η], d[:P], d[:ts], d[:T]; penalty_factor=corr.penalty_factor, maxtime=corr.maxtime)
end

function optimized_independent_simple_correction(ramp, ηs, P, ts, T; penalty_factor, maxtime, alg=Fminbox(NelderMead()))
    H = ham_with_corrections
    results = Vector{Float64}[]
    function cost_function(x::Vector, t)
        ham = Hermitian(H((ramp, ηs, IndependentSimpleCorrection(x), P), t))
        vals = eigvals(ham)
        return vals[2] - vals[1]
    end

    lambda_limit = 100
    middle = optimize(x -> cost_function(x, T / 2), lambda_limit .* [-1, -1], lambda_limit * [1, 1], [1.0, 1.0], alg, Optim.Options(time_limit=maxtime, f_abstol=1e-14, f_reltol=1e-14))
    middle_result = iszero(middle.minimizer) ? [1, 1] / sqrt(2) : middle.minimizer / norm(middle.minimizer)
    # println(middle_result)
    # initial = [0.0]
    for t in ts
        initial = length(results) > 0 ? norm(results[end]) : 0.0
        result = optimize(x -> cost_function(x .* middle_result, t) + penalty_factor * abs2(initial - x), -lambda_limit, lambda_limit, Brent(); abs_tol=1e-14, rel_tol=1e-14, time_limit=maxtime / length(ts))
        push!(results, only(result.minimizer) .* middle_result)
    end
    return IndependentSimpleCorrection(linear_interpolation(ts, results))
end
