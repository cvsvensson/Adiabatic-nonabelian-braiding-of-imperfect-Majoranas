
## Define a new braiding_deltas function that varies the plateau length
# The down time can be different from the up time

function braiding_deltas_new(t, T, Δmin, Δmax, k, inverted, args...)
    Δ1max = Δmax / 1
    Δ2max = Δmax / 1
    Δ3max = Δmax
    pulse_length = T / 2
    Δ1 = smooth_pulse(t, 0, pulse_length, Δmin, Δ1max, k) + smooth_pulse(t, T, pulse_length, Δmin, Δ1max, k) + smooth_pulse(t, 2T, pulse_length, Δmin, Δ1max, k)
    Δ2 = smooth_pulse(t, T/3, pulse_length, Δmin, Δ2max, k) + smooth_pulse(t, T + T/3, pulse_length, Δmin, Δ2max, k)
    Δ3 = smooth_pulse(t, 2T/3, pulse_length, Δmin, Δ3max, k) + smooth_pulse(t, T + 2T/3, pulse_length, Δmin, Δ3max, k)
    if inverted
        # Δ1 = inverted_pulse(t, 0, pulse_length, Δmin, Δ1max, k) + inverted_pulse(t, T, pulse_length, Δmin, Δ1max, k) + inverted_pulse(t, 2T, pulse_length, Δmin, Δ1max, k)
        Δ2 = inverted_pulse(t, T/3, pulse_length, Δmin, Δ2max, k) + inverted_pulse(t, T + T/3, pulse_length, Δmin, Δ2max, k)
        Δ3 = inverted_pulse(t, 2T/3, pulse_length, Δmin, Δ3max, k) + inverted_pulse(t, T + 2T/3, pulse_length, Δmin, Δ3max, k)
    end
    return Δ1, Δ2, Δ3
end

# Define the sigmoid function that respresents a smooth
function sigmoid(x, k)
    y = 1 / (1 + exp(-k*x))
    return y
end

# Add two sigmoid functions to create a smooth pulse centered at Tcenter with a duration of Tup
function smooth_pulse(t, Tcenter, Tup, Δmin, Δmax, k)
    Δ = Δmin + (Δmax - Δmin) * (sigmoid(t - Tcenter +Tup/2, k) ) * (1 - sigmoid(t - Tcenter - Tup/2, k))
    return Δ
end

# Define puls that switches sign after half the time
function inverted_pulse(t, Tcenter, Tup, Δmin, Δmax, k)
    Δ = Δmin + (Δmax - Δmin) *
        (sigmoid(t - Tcenter + Tup/2, k) - sigmoid(t - Tcenter - Tup/2, k)) *
        (1 - 2 * sigmoid(t - Tcenter, k))

    # Change the above definition to invert pulse sign after 3/4 of the time
    Δ = Δmin + (Δmax - Δmin) *
        (sigmoid(t - Tcenter + Tup/2, k) - sigmoid(t - Tcenter - Tup/2, k)) *
        (1 - 2 * sigmoid(t - Tcenter  - Tup/3, k))
    return Δ
    
end
