function drho(u, p, t)
    ham = H(p, t)
    return 1im * ham * u
end
function drho!(du, u, (p, Hcache), t)
    ham = H!(Hcache, p, t)
    mul!(du, ham, u, 1im, 0)
    return du
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
    transformed_matrices = map(mtype, values(Pnew.dict))
    newdict = Majoranas.OrderedDict(zip(keys(Pnew.dict), transformed_matrices))
    return HamiltonianBasis(newdict, γ.fermion_basis)
end

function remove_from_basis(remove_labels, basis)
    new_dict = copy(basis.dict)
    for label in remove_labels
        delete!(new_dict, label)
    end
    return HamiltonianBasis(new_dict, basis.fermion_basis) # returns undef on removed labels...
end

function parity_operators_old(nbr_of_majoranas, majorana_labels, mtype)
    N = div(nbr_of_majoranas, 2)
    c = FermionBasis(1:N, qn=QuantumDots.parity)
    γ = MajoranaWrapper(c, majorana_labels)
    return parity_operators(γ, p -> (mtype(p[2^(N-1)+1:end, 2^(N-1)+1:end])))
end

get_op(H, p) = MatrixOperator(H(p, 0, 1im); update_func=(A, u, p, t) -> H(p, t, 1im))
get_op(H, H!, p) = MatrixOperator(H(p, 0, 1im); update_func=(A, u, p, t) -> H(p, t, 1im), (update_func!)=(A, u, p, t) -> H!(A, p, t, 1im))

function get_iH_interpolation(H, p, ts)
    cubic_spline_interpolation(ts, [H(p, t, 1im) for t in ts], extrapolation_bc = Periodic())
end
get_iH_interpolation_op(H, p, ts) = get_op_from_interpolation(get_iH_interpolation(H, p, ts))
get_op_from_interpolation(int) = MatrixOperator(int(0.0); update_func=(A, u, p, t) -> int(t))


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
