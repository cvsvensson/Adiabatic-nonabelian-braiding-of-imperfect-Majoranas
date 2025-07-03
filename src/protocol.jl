const MajoranaLabels = [:M, :M̃, :L, :L̃, :R, :R̃]
function parity_operators(totalparity, mtype)
    @majoranas γ
    H = FermionicHilbertSpaces.majorana_hilbert_space(MajoranaLabels, ParityConservation(totalparity))
    Dict((l1, l2) => mtype(matrix_representation(1im * γ[l1] * γ[l2], H)) for (l1, l2) in Base.product(MajoranaLabels, MajoranaLabels) if l1 != l2)
end

struct RampProtocol{T,F}
    total_time::T
    smooth_step::F
    function RampProtocol(total_time::T, smooth_step::F) where {T,F}
        new{T,F}(total_time, smooth_step)
    end
end

RampProtocol(T, k::Number) = RampProtocol(T, smooth_step(k))
smooth_step(k, x) = 1 / 2 + tanh(k * x) / 2
smooth_step(k) = Base.Fix1(smooth_step, k)
function (ramp::RampProtocol)(t)
    T = ramp.total_time
    f = ramp.smooth_step
    ρ1 = f(cos(2pi * (t - 0) / T))
    ρ2 = f(cos(2pi * (t - T / 3) / T))
    ρ3 = f(cos(2pi * (t - 2T / 3) / T))
    return (ρ1, ρ2, ρ3) ./ sqrt(ρ1^2 + ρ2^2 + ρ3^2) # normalize
end

function setup_problem(dict)
    @unpack η, T, k, steps, correction, totalparity = dict
    N = 3
    d = 2^(N - 1)
    _u0 = dict[:initial]
    mtype = get(dict, :mtype, SMatrix{d,d,ComplexF64})
    vtype = get(dict, :vtype, SVector{d,ComplexF64})
    P = parity_operators(totalparity, mtype)
    ramp = RampProtocol(T, k)
    tspan = (0.0, 2 * T)
    ts = range(0, tspan[2], steps)
    newdict = Dict(dict..., :ramp => ramp, :ts => ts, :tspan => tspan, :P => P)
    corr = setup_correction(correction, newdict)
    p = (ramp, η, corr, P)
    interpolate = get(dict, :interpolate_corrected_hamiltonian, false)
    H(p, t) = ham_with_corrections(p, t)
    op = interpolate ? get_iH_interpolation_op(ham_with_corrections, p, ts) : get_op(ham_with_corrections, p)
    u0 = process_initial_state(_u0, P, (mtype, vtype))
    prob = ODEProblem{false}(op, u0, tspan, p)
    return Dict(newdict..., :correction => corr, :p => p, :op => op, :odeprob => prob, :H => H, :u0 => _u0)
end
function OrdinaryDiffEqCore.solve(prob::Dict, alg=Tsit5(); abstol=1e-6, reltol=1e-6, saveat=range(0, 2prob[:T], 200), kwargs...)
    solve(prob[:odeprob], alg; abstol, reltol, saveat, kwargs...)
end
function process_initial_state(u0::Pair{<:Tuple,Int}, P, (mtype, vtype))
    label = first(u0)
    parity = last(u0)
    vtype(collect(first(eachcol(eigen(Hermitian(-parity * P[label] + P[:M, :M̃]), 1:1).vectors))))
end
function process_initial_state(::UniformScaling, P, (mtype, vtype))
    mtype(I)
end
