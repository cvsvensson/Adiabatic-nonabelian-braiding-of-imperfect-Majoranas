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