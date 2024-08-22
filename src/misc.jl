
function drho(u, p, t)
    ham = H(p, t)
    return 1im * ham * u
end
function drho!(du, u, (p, Hcache), t)
    ham = H!(Hcache, p, t)
    mul!(du, ham, u, 1im, 0)
    return du
end

function parity_operators(γ::MajoranaWrapper, transform=Matrix)
    Dict([(k1, k2) => transform(1.0im * γ[k1] * γ[k2]) for k1 in keys(γ.majoranas), k2 in keys(γ.majoranas)])
end

function matrix_vec_types(use_static_arrays, inplace, N)
    if use_static_arrays && inplace
        return MMatrix{2^(N - 1),2^(N - 1)}, MVector{2^(N - 1)}
    elseif use_static_arrays && !inplace
        return SMatrix{2^(N - 1),2^(N - 1)}, SVector{2^(N - 1)}
    end
    return Matrix, Vector
end

function parity_operators(γ::SingleParticleMajoranaBasis, parity, mtype)
    Pnew = ProjectedHamiltonianBasis(γ, parity)
    transformed_matrices = map(matrix -> mtype(matrix), Pnew.dict.values)
    return HamiltonianBasis(QuantumDots.Dictionary(Pnew.dict.indices, transformed_matrices), γ.fermion_basis)
end

function parity_operators_old(nbr_of_majoranas, majorana_labels, mtype)
    N = div(nbr_of_majoranas, 2)
    c = FermionBasis(1:N, qn=QuantumDots.parity)
    γ = MajoranaWrapper(c, majorana_labels)
    return parity_operators(γ, p -> (mtype(p[2^(N-1)+1:end, 2^(N-1)+1:end])));
end


get_op(H, H!, p) = MatrixOperator(H(p, 0, 1im); update_func=(A, u, p, t) -> H(p, t, 1im), (update_func!)=(A, u, p, t) -> H!(A, p, t, 1im))

@testitem "Test old vs new parities" begin
    using Majoranas
    nbr_of_majoranas = 6
    N = div(nbr_of_majoranas, 2)
    majorana_labels = 0:5
    γ = SingleParticleMajoranaBasis(nbr_of_majoranas, majorana_labels)
    parity = 1
    use_static_arrays = true
    inplace = !use_static_arrays
    mtype, _ = MajoranaBraiding.matrix_vec_types(use_static_arrays, inplace, N)
    Pnew = parity_operators(γ, parity, mtype)
    Pold = MajoranaBraiding.parity_operators_old(nbr_of_majoranas, majorana_labels, mtype)
    ## Get the majoranas
    @test all([Pold[lab...] == Pnew[lab...] for lab in labels(Pnew)[1:end-1]])
end
