function drho(u, p, t)
    ham = H(p, t)
    return 1im * ham * u
end
function drho!(du, u, (p, Hcache), t)
    ham = H!(Hcache, p, t)
    mul!(du, ham, u, 1im, 0)
    return du
end

get_op(H, p, T) = MatrixOperator(H(p, 0, T * 1im); update_func=(A, u, p, t) -> H(p, t, 1im))
get_op(H, H!, p, T) = MatrixOperator(H(p, 0, T * 1im); update_func=(A, u, p, t) -> H(p, t, T * 1im), (update_func!)=(A, u, p, t) -> H!(A, p, t, T * 1im))

function get_iH_interpolation(H, p, ts, T)
    cubic_spline_interpolation(ts, [H(p, t, T * 1im) for t in ts], extrapolation_bc=Periodic())
end
get_iH_interpolation_op(H, p, ts, T) = get_op_from_interpolation(get_iH_interpolation(H, p, ts, T))
get_op_from_interpolation(int) = MatrixOperator(int(0.0); update_func=(A, u, p, t) -> int(t))

