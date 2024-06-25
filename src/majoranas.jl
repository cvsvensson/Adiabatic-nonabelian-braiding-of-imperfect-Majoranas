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
