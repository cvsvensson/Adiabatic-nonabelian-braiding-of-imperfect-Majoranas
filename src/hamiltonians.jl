
ham_with_corrections(p, t, α=1) = _ham_with_corrections(p..., t, α)

function _ham_with_corrections(ramp, ϵs, ζs, correction, P, t, α=1)
    Δs = ramp(t)
    Ham = (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
           ϵs[1] * P[0, 1] + ϵs[2] * P[2, 4] + ϵs[3] * P[3, 5] +
           _error_ham(Δs, ζs, P))
    Ham += correction(t, Δs, ζs, P, Ham)
    return Ham * α
end

_error_ham(Δs, ζs, P) = +Δs[2] * ζs[1] * ζs[2] * P[1, 4] + Δs[3] * ζs[1] * ζs[3] * P[1, 5]
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


struct EigenEnergyCorrection{B,T} <: AbstractCorrection
    basis::B
    scaling::T
    function EigenEnergyCorrection(basis, scaling)
        newscaling = _process_constant_scaling(scaling)
        new{typeof(basis), typeof(newscaling)}(basis, newscaling)
    end
end
EigenEnergyCorrection(basis) = EigenEnergyCorrection(basis, t -> true)
(corr::EigenEnergyCorrection)(t, Δs, ζs, P, ham) = iszero(corr.scaling(t)) ? zero(ham) : (corr.scaling(t) * full_energy_correction_term(ham, corr.basis))
Base.isless(::EigenEnergyCorrection, ::AbstractCorrection) = false
Base.isless(::AbstractCorrection, ::EigenEnergyCorrection) = true

function full_energy_correction_term(ham, basis; alg=Majoranas.WM_BACKSLASH())
    vals, vecs = eigen(Hermitian(ham))
    δE = (vals[2] - vals[1]) / 2
    # push the lowest energy states δE closer together
    weak_ham_prob = WeakMajoranaProblem(basis, vecs, nothing, [nothing, nothing, nothing, δE])
    sol = solve(weak_ham_prob, alg)
    #=δv = (vecs[:, 1] * vecs[:, 1]' - vecs[:, 2] * vecs[:, 2]') + =#
    #=     (vecs[:, 3] * vecs[:, 3]' - vecs[:, 4] * vecs[:, 4]')=#
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

function analytical_exact_simple_correction(ζ, ramp, ts, totalparity=1)
    results = Float64[]
    for t in ts
        # Find roots of the energy split function
        initial = length(results) > 0 ? results[end] : 0.0
        initial = 0.0
        result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity)
        push!(results, result)
    end
    return SimpleCorrection(linear_interpolation(ts, results))
end
function find_zero_energy_from_analytics(ζ, ramp, t, initial=0.0, totalparity=1)
    result = find_zero(x -> energy_split(x, ζ, ramp, t, totalparity), initial)
    return result
end
function energy_split(x, ζ, ramp, t, totalparity=1)
    Η, Λ = energy_parameters(x, ζ, ramp, t)
    μ, α, β, ν = groundstate_components(x, ζ, ramp, t) 

    Δϵ = β * ν + Η * μ * α + Λ * α * ν + x * sign(totalparity) 
    return Δϵ
end
function energy_parameters(x, ζ, ramp, t)
    Δs = ramp(t)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ = √(Δs[1]^2 + Δs[2]^2 + Δs[3]^2)
    ρ = Δ23 / Δ
    η = -ζ^2

    Η = η * ρ^2 + x * √(1 - ρ^2)
    Λ = ρ * x - ρ * √(1 - ρ^2) * η
    return [Η, Λ]
end
function groundstate_components(x, ζ, ramp, t)
    Η, Λ = energy_parameters(x, ζ, ramp, t)
    
    θ_μ = -1/2 * atan(2 * Λ * Η, 1 + Λ^2 - Η^2)
    μ = cos(θ_μ)
    ν = sin(θ_μ)
    
    θ_α = atan(Η * tan(θ_μ) - Λ)
    α = cos(θ_α)
    β = sin(θ_α)
    return [μ, α, β, ν]
end

function single_braid_gate_improved(P, ζ, ramp, T, totalparity=1)
    θ_α, θ_μ = single_braid_gate_analytical_angle(P, ζ, ramp, T, totalparity)

    U_12 = exp(+1im * (1 * π/4 + 0 * θ_μ/2) * 0 * P[1, 2] + 0 * 1im * θ_α/2 * P[0, 4])  # Deactivated via 0 *
    U_23 = exp(+1im * π/4 * (cos(θ_α)^2 ) * P[2, 3] + 1im * π/4 * sin(θ_μ)^2 * P[4, 5]
                + 0 * 1im * 1/2 * cos(θ_μ) * sin(θ_μ) * (P[1, 3] - P[1, 2])             # Deactivated via 0 *
                + 0 * 1im * 1/2 * cos(θ_α) * sin(θ_α) * (P[0, 5] - P[0, 4]) )           # Deactivated via 0 *
    U_31 = exp(-1im * (1 * π/4 + 0 * θ_μ/2) * 0 * P[1, 3] - 0 * 1im * θ_α/2 * P[0, 5])  # Deactivated via 0 *
    # return matrix multiplication of the three unitaries
    return U_31 * U_23 * U_12
    #return ( cos(θ) * unit + sin(θ) *  (1im) * P[2, 3] ) * (cos(ϕ) * unit + sin(ϕ) * (1im) * P[4, 5])
end

function single_braid_gate_lucky_guess(P, ζ, ramp, T, totalparity=1)
    θ_α, θ_μ = single_braid_gate_analytical_angle(P, ζ, ramp, T, totalparity)
    
    α = cos(θ_α)
    ν = sin(θ_μ)

    U = exp(1im * π/4 * (α - ν) * P[2, 3])
    return U
end

function single_braid_gate_analytical_angle(P, ζ, ramp, T, totalparity=1)
    t = T/2
    initial = 0.0
    result = find_zero_energy_from_analytics(ζ, ramp, t, initial, totalparity)
    μ, α, β, ν = MajoranaBraiding.groundstate_components(result, ζ, ramp, t)
    #θ = atan(μ, ν) / 2
    #ϕ = -atan(β, α) /2

    θ_α = atan(β, α)
    θ_μ = atan(ν, μ)
    
    return θ_α, θ_μ
end

function single_braid_gate_fit(ω)
    return exp(1im * ω * P[2, 3])
end

function braid_gate_prediction(gate, ω)
    prediction = single_braid_gate_fit(ω)

    proj = Diagonal([1, 0, 0, 1])
    proj = Diagonal([0, 1, 1, 0])
    single_braid_fidelity = gate_fidelity(proj * prediction * proj, proj * gate * proj)  
    return single_braid_fidelity
end

function braid_gate_best_angle(gate)
    # ω = find_zero(ω -> 1-braid_gate_prediction(gate, ω), 0.0)
    # I want to rewrite the above line
    # Instead of a true zero I want to search a minimum of the function abs(1 - braid_gate_prediction(gate, ω))
    # I want to use Optim.optimize to do this
    ω = optimize(ω -> abs(1 - braid_gate_prediction(gate, ω)), 0.0, π/2).minimizer
    return ω, braid_gate_prediction(gate, ω)
end