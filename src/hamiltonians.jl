function drho(u, p, t)
    ham = H(p, t)
    return 1im * ham * u
end
function drho!(du, u, (p, Hcache), t)
    ham = H!(Hcache, p, t)
    mul!(du, ham, u, 1im, 0)
    return du
end

get_op(H, p, T) = MatrixOperator(H(p, 0, T * 1im); update_func=(A, u, p, t) -> H(p, t, 1im))
get_op(H, H!, p, T) = MatrixOperator(H(p, 0, T * 1im); update_func=(A, u, p, t) -> H(p, t, T * 1im), (update_func!)=(A, u, p, t) -> H!(A, p, t, T * 1im))

function get_iH_interpolation(H, p, ts, T)
    cubic_spline_interpolation(ts, [H(p, t, T * 1im) for t in ts], extrapolation_bc=Periodic())
end
get_iH_interpolation_op(H, p, ts, T) = get_op_from_interpolation(get_iH_interpolation(H, p, ts, T))
get_op_from_interpolation(int) = MatrixOperator(int(0.0); update_func=(A, u, p, t) -> int(t))

ham_with_corrections(p, t, α=1) = _ham_with_corrections(p..., t, α)
_ham_with_corrections(η::Number, k, gapscaling, correction, P, totalparity, t, α) = _ham_with_corrections((η, η, η), k, gapscaling, correction, P, totalparity, t, α)
function _ham_with_corrections(ηs, k, gapscaling, correction, P, totalparity, t, α)
    ρs = get_rhos(k, t)
    Ham = (ρs[1] * P[:M, :M̃] +
           ρs[2] * P[:M, :L] +
           ρs[3] * P[:M, :R] +
           # errors
           ρs[2] * sqrt(ηs[1] * ηs[2]) * P[:M̃, :L̃] +
           -totalparity * ρs[3] * sqrt(ηs[1] * ηs[3]) * P[:M̃, :R̃])
    Ham += correction(t, ρs, ηs, P, totalparity, Ham)
    return Ham * α * gapscaling(t)
end

abstract type AbstractCorrection end
(corr::AbstractCorrection)(t, ρs, ηs, P, totalparity, ham) = error("(corr::C)(t, ρs, ηs, P,totalparity, ham) not implemented for C=$(typeof(corr))")
struct NoCorrection <: AbstractCorrection end
(corr::NoCorrection)(t, ρs, ηs, P, totalparity, ham) = 0I
struct SimpleCorrection{T} <: AbstractCorrection
    scaling::T
end
setup_correction(::NoCorrection, ::Dict) = NoCorrection()

SimpleCorrection() = SimpleCorrection(true)
SimpleCorrection(scaling::Number) = SimpleCorrection(t -> scaling)
(corr::SimpleCorrection)(t, ρs, ηs, P, totalparity, ham) = corr.scaling(t) * (P[:L, :L̃] - totalparity * P[:R, :R̃])
setup_correction(corr::SimpleCorrection, ::Dict) = corr

struct IndependentSimpleCorrection{T} <: AbstractCorrection
    scaling::T
end
function IndependentSimpleCorrection(scaling1, scaling2)
    IndependentSimpleCorrection(t -> (scaling1(t), scaling2(t)))
end
function IndependentSimpleCorrection(scaling1::Number, scaling2::Number)
    IndependentSimpleCorrection(t -> (scaling1, scaling2))
end
IndependentSimpleCorrection(scalings::Vector{<:Number}) = length(scalings) == 2 ? IndependentSimpleCorrection(scalings...) : error("scalings must be a vector of length 2")
setup_correction(corr::IndependentSimpleCorrection, ::Dict) = corr

function (corr::IndependentSimpleCorrection)(t, Δs, ηs, P, totalparity, ham)
    s = corr.scaling(t)
    s[1] * P[:L, :L̃] - totalparity * s[2] * P[:R, :R̃]
end

struct OptimizedSimpleCorrection <: AbstractCorrection end

function setup_correction(::OptimizedSimpleCorrection, d::Dict)
    return optimized_simple_correction(d[:η], d[:k], d[:P], d[:totalparity], d[:ts])
end
function optimized_simple_correction(ηs, k, P, totalparity, ts; alg=BFGS())
    H = ham_with_corrections
    results = Float64[]
    function cost_function(x, t)
        vals = eigvals(H((ηs, k, t -> 1, SimpleCorrection(x), P, totalparity), t))
        return vals[2] - vals[1]
    end
    for t in ts
        f(x) = cost_function(only(x), t)
        initial = length(results) > 0 ? results[end] : 0.0
        result = optimize(f, [initial], alg, Optim.Options(time_limit=10 / length(ts)))
        push!(results, only(result.minimizer))
    end
    return SimpleCorrection(linear_interpolation(ts, results))
end

struct OptimizedIndependentSimpleCorrection <: AbstractCorrection
    maxtime::Float64
    penalty_factor::Float64
end

function setup_correction(corr::OptimizedIndependentSimpleCorrection, d::Dict)
    return optimized_independent_simple_correction(d[:η], d[:k], d[:P], d[:totalparity], d[:ts]; penalty_factor=corr.penalty_factor, maxtime=corr.maxtime)
end

function optimized_independent_simple_correction(ηs, k, P, totalparity, ts; penalty_factor, maxtime, alg=Fminbox(NelderMead()))
    H = ham_with_corrections
    results = Vector{Float64}[]
    function cost_function(x::Vector, t)
        ham = Hermitian(H((ηs, k, t -> 1, IndependentSimpleCorrection(x), P, totalparity), t))
        vals = eigvals(ham)
        return vals[2] - vals[1]
    end

    lambda_limit = 100
    middle = optimize(x -> cost_function(x, 1 / 2), lambda_limit .* [-1, -1], lambda_limit * [1, 1], [1.0, 1.0], alg, Optim.Options(time_limit=maxtime, f_abstol=1e-14, f_reltol=1e-14))
    middle_result = iszero(middle.minimizer) ? [1, 1] / sqrt(2) : middle.minimizer / norm(middle.minimizer)
    for t in ts
        initial = length(results) > 0 ? norm(results[end]) : 0.0
        result = optimize(x -> cost_function(x .* middle_result, t) + penalty_factor * abs2(initial - x), -lambda_limit, lambda_limit, Brent(); abs_tol=1e-14, rel_tol=1e-14, time_limit=maxtime / length(ts))
        push!(results, only(result.minimizer) .* middle_result)
    end
    return IndependentSimpleCorrection(linear_interpolation(ts, results))
end
