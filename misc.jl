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
