struct RampProtocol{Mi,Ma,T,F}
    delta_mins::Mi
    delta_maxes::Ma
    total_time::T
    smooth_step::F
    extra_shifts::SVector{3,Float64}
    function RampProtocol(Δmin::Mi, Δmax::Ma, total_time::T, smooth_step::F, extra_shifts=@SVector [0, 0, 0]) where {Mi,Ma,T,F}
        Δmin = process_delta(Δmin)
        Δmax = process_delta(Δmax)
        new{typeof(Δmin),typeof(Δmax),T,F}(Δmin, Δmax, total_time, smooth_step, extra_shifts)
    end
end

RampProtocol(Δmin, Δmax, T, k::Number, extra_shifts=@SVector [0, 0, 0]) = RampProtocol(Δmin, Δmax, T, smooth_step(k), extra_shifts)
process_delta(Δ::Number) = Δ .* [1, 1, 1]
process_delta(Δ) = Δ
smooth_step(k, x) = 1 / 2 + tanh(k * x) / 2
smooth_step(k) = Base.Fix1(smooth_step, k)

@inbounds function get_deltas(p::RampProtocol, t)
    T = p.total_time
    f = p.smooth_step
    Δmin = p.delta_mins
    Δmax = p.delta_maxes
    extra_shifts = p.extra_shifts * T
    shifts = (@SVector [0.0, T / 3, 2T / 3]) + extra_shifts
    fi(i) = Δmin[i] + (Δmax[i] - Δmin[i]) * f(cos(2pi * (t - shifts[i]) / T))
    ntuple(fi, Val(3))
end

(ramp::RampProtocol)(t) = get_deltas(ramp, t)

function setup_problem(dict)
    @unpack ζ, Δmin, Δmax, T, k, steps, correction, totalparity = dict
    N = 3
    d = 2^(N - 1)
    _u0 = dict[:initial]
    mtype = get(dict, :mtype, SMatrix{d,d,ComplexF64})
    vtype = get(dict, :vtype, SVector{d,ComplexF64})
    P = parity_operators(totalparity, mtype)
    extra_shifts = get(dict, :extra_shifts, @SVector [0, 0, 0])
    ramp = RampProtocol(Δmin, Δmax, T, k, extra_shifts)
    tspan = (0.0, 2 * T)
    ts = range(0, tspan[2], steps)
    newdict = Dict(dict..., :ramp => ramp, :ts => ts, :tspan => tspan, :P => P)
    corr = setup_correction(correction, newdict)
    p = (ramp, ζ, corr, P)
    interpolate = get(dict, :interpolate_corrected_hamiltonian, false)
    H(p, t) = ham_with_corrections(p, t)
    op = interpolate ? get_iH_interpolation_op(ham_with_corrections, p, ts) : get_op(ham_with_corrections, p)
    u0 = process_initial_state(_u0, P, (mtype, vtype))
    prob = ODEProblem{false}(op, u0, tspan, p)
    return Dict(newdict..., :correction => corr, :p => p, :op => op, :odeprob => prob, :H => H, :u0 => _u0)
end
function process_initial_state(u0::Pair{<:Tuple,Int}, P, (mtype, vtype))
    label = first(u0)
    parity = last(u0)
    vtype(collect(first(eachcol(eigen(Hermitian(-parity * P[label] + P[:M, :M̃]), 1:1).vectors))))
end
function process_initial_state(::UniformScaling, P, (mtype, vtype))
    mtype(I)
end

const MajoranaLabels = [:M, :M̃, :L, :L̃, :R, :R̃]
function parity_operators(totalparity, mtype)
    @majoranas γ
    H = FermionicHilbertSpaces.majorana_hilbert_space(MajoranaLabels, ParityConservation(totalparity))
    Dict((l1, l2) => mtype(matrix_representation(1im * γ[l1] * γ[l2], H)) for (l1, l2) in Base.product(MajoranaLabels, MajoranaLabels) if l1 != l2)
end