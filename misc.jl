struct MajoranaWrapper{B,D}
    basis::B
    majoranas::D
    function MajoranaWrapper(basis, labels=collect(Base.product(keys(basis), (:a, :b))))
        N = length(basis)
        length(labels) == 2 * N || throw(ErrorException("Number of majoranas is not twice the fermion number"))
        majA = map(f -> f + f', basis)
        majB = map(f -> 1im * (f - f'), basis)
        dA = QuantumDots.dictionary(zip(labels[1:N], values(majA)))
        dB = QuantumDots.dictionary(zip(labels[N+1:2N], values(majB)))
        d = merge(dA, dB)
        new{typeof(basis),typeof(d)}(basis, d)
    end
end
Base.getindex(g::MajoranaWrapper, i...) = g.majoranas[i...]

smooth_step(x, k) = 1 / 2 + tanh(k * x) / 2

function drho!(du, u, p, t)
    ham = H(p, t)
    mul!(du, ham, u, 1im, 0)
    return du
end

function parity_operators(γ::MajoranaWrapper)
    Dict([(k1, k2) => Matrix(1.0im * γ[k1] * γ[k2]) for k1 in keys(γ.majoranas), k2 in keys(γ.majoranas)])
end

## Give the value of the three deltas at time t in the three point majorana braiding protocol
function braiding_deltas(t, T, Δmin, Δmax, k, args...)
    Δ1 = Δtrajectory(t, T, Δmin, Δmax, k)
    Δ2 = Δtrajectory(t + T / 3, T, Δmin, Δmax, k)
    Δ3 = Δtrajectory(t + 2T / 3, T, Δmin, Δmax, k)
    return Δ1, Δ2, Δ3
end
function Δtrajectory(t, T, Δmin, Δmax, k, args...)
    dΔ = Δmax - Δmin
    tp = mod(t, T) - 3T / 12
    Δmin + dΔ * smooth_step(tp - 1T / 6, k) - dΔ * smooth_step(tp - 4T / 6, k)
end
