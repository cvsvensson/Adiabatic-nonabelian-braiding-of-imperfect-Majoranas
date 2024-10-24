# Smooth pulse function definition
function smooth_pulse(t, t_0, delta_t, k)
    left_transition = step((t - (t_0 - delta_t / 2)) / k)
    right_transition = step((t - (t_0 + delta_t / 2)) / k)
    return 0.5 * (left_transition - right_transition)
end

function sigmoid(x)
    # Q: What alternatives are there to tanh?
    # A: erf, sigmoid, heaviside, etc.
    # Q: What about arctan?
    # A: arctan is not a good choice because it is not smooth
    # I still want to try arctan
    return tanh(x)
    #return 2*atan(x)/π + 1
    #return -1 / (1 + exp(x))
end

function step(x)
    return sigmoid(x)
end

# RampProtocol struct with delta_t and k directly included
struct RampProtocol{Mi, Ma, T, K, Dt}
    delta_mins::Mi
    delta_maxes::Ma
    total_time::T
    smoothness::K       # k is stored as smoothness
    delta_t::Dt
    function RampProtocol(Δmin::Mi, Δmax::Ma, total_time::T, k::K, delta_t::Dt) where {Mi, Ma, T, K, Dt}
        if length(Δmin) != 3 || length(Δmax) != 3
            throw(ArgumentError("Δmin and Δmax must have length 3"))
        end
        new{Mi, Ma, T, K, Dt}(Δmin, Δmax, total_time, k, delta_t)
    end
end

# Overloaded constructor that directly accepts k as a number
#RampProtocol(Δmin, Δmax, T, k::Number, delta_t) = RampProtocol(Δmin, Δmax, T, k, delta_t)

# Modified get_deltas to use smooth_pulse with k passed directly
@inbounds function get_deltas(p::RampProtocol, t)
    T = p.total_time
    Δmin = p.delta_mins
    Δmax = p.delta_maxes
    delta_t = p.delta_t
    k = p.smoothness    # k is now passed directly as a scalar
    shifts = @SVector [0.0, T / 3, 2T / 3]

    # Use smooth_pulse for each element
    fi(i) = Δmin[i] + (Δmax[i] - Δmin[i]) *
                (smooth_pulse(t, shifts[i] + 0*T, delta_t, k) +
                 smooth_pulse(t, shifts[i] + 1*T, delta_t, k) +
                    smooth_pulse(t, shifts[i] + 2*T, delta_t, k))

    # Return tuple of 3 elements
    ntuple(fi, Val(3))
end

# Make the RampProtocol callable
(ramp::RampProtocol)(t) = get_deltas(ramp, t)