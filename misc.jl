struct MajoranaWrapper{B,D}
    basis::B
    majoranas::D
    function MajoranaWrapper(basis, labels=collect(Base.product(keys(basis), (:a, :b))))
        N = length(basis)
        length(labels) == 2 * N || throw(ErrorException("Number of majoranas is not twice the fermion number"))
        majs = vec(reduce(hcat, [[f + f', 1im * (f - f')] for f in basis]))
        d = QuantumDots.dictionary(zip(labels, values(majs)))
        new{typeof(basis),typeof(d)}(basis, d)
    end
end
Base.getindex(g::MajoranaWrapper, i...) = g.majoranas[i...]

function MajoranaBasis(labels; qn=QuantumDots.parity)
    N = length(labels)
    iseven(N) || throw(ErrorException("Number of majoranas must be even"))
    c = FermionBasis(1:div(N, 2); qn)
    MajoranaWrapper(c, labels)
end

smooth_step(x, k) = 1 / 2 + tanh(k * x) / 2

function drho(u, p, t)
    ham = H(p, t)
    return 1im * ham * u
end
function drho!(du, u, (p, Hcache), t)
    ham = H!(Hcache, p, t)
    mul!(du, ham, u, 1im, 0)
    return du
end

function parity_operators(γ::MajoranaWrapper)
    Dict([(k1, k2) => Matrix(1.0im * γ[k1] * γ[k2]) for k1 in keys(γ.majoranas), k2 in keys(γ.majoranas)])
end

## Give the value of the three deltas at time t in the three point majorana braiding protocol
function braiding_deltas(t, T, Δmin, Δmax, k, args...)
    Δ1 = Δtrajectory(t, T, Δmin, Δmax / 3, k)
    Δ2 = Δtrajectory(t - T / 3, T, Δmin, Δmax / 2, k)
    Δ3 = Δtrajectory(t - 2T / 3, T, Δmin, Δmax, k)
    return Δ1, Δ2, Δ3
end
function Δtrajectory(t, T, Δmin, Δmax, k, args...)
    dΔ = Δmax - Δmin
    Δmin + dΔ * smooth_step(cos(2pi * t / T), k)
end

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