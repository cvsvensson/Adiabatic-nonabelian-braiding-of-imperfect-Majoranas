
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


get_op(H, p) = MatrixOperator(H(p, 0, 1im); update_func=(A, u, p, t) -> H(p, t, 1im))
get_op(H, H!, p) = MatrixOperator(H(p, 0, 1im); update_func=(A, u, p, t) -> H(p, t, 1im), (update_func!)=(A, u, p, t) -> H!(A, p, t, 1im))