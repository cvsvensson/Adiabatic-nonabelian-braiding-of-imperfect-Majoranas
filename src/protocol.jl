const MajoranaLabels = [:M, :M̃, :L, :L̃, :R, :R̃]
function parity_operators(totalparity, mtype)
    @majoranas γ
    H = majorana_hilbert_space(MajoranaLabels, ParityConservation(totalparity))
    Dict((l1, l2) => mtype(matrix_representation(1im * γ[l1] * γ[l2], H)) for (l1, l2) in Base.product(MajoranaLabels, MajoranaLabels) if l1 != l2)
end

smooth_step(k, x) = 1 / 2 + tanh(k * x) / 2
smooth_step(k) = Base.Fix1(smooth_step, k)
function get_rhos(k, t)
    if t > 2
        @warn "t should be in [0, 2], got $t"
    end
    f = smooth_step(k)
    ρ1 = f(cos(2pi * t))
    ρ2 = f(cos(2pi * (t - 1 / 3)))
    ρ3 = f(cos(2pi * (t - 2 / 3)))
    return (ρ1, ρ2, ρ3) ./ sqrt(ρ1^2 + ρ2^2 + ρ3^2) # normalize
end

function setup_problem(dict)
    @unpack η, T, k, steps, correction, totalparity = dict
    N = 3
    d = 2^(N - 1)
    gapscaling = get(dict, :gapscaling, t -> 1)
    mtype = get(dict, :mtype, SMatrix{d,d,ComplexF64})
    vtype = get(dict, :vtype, SVector{d,ComplexF64})
    P = parity_operators(totalparity, mtype)
    tspan = (0.0, 2.0)
    ts = range(tspan..., steps)
    newdict = Dict(dict..., :ts => ts, :tspan => tspan, :P => P, :gapscaling => gapscaling)
    corr = setup_correction(correction, newdict)
    p = (η, k, gapscaling, corr, P)
    interpolate = get(dict, :interpolate_corrected_hamiltonian, false)
    H(p, t) = ham_with_corrections(p, t, T)
    op = interpolate ? get_iH_interpolation_op(ham_with_corrections, p, ts, T) : get_op(ham_with_corrections, p, T)
    u0 = process_initial_state(dict[:initial], P, (mtype, vtype))
    prob = ODEProblem{false}(op, u0, tspan, p)
    return Dict(newdict..., :correction => corr, :p => p, :op => op, :odeprob => prob, :H => H, :u0 => u0)
end

function OrdinaryDiffEqCore.solve(prob::Dict, alg=Tsit5(); abstol=1e-6, reltol=1e-6, saveat=range(0, 2, 100), kwargs...)
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
