
#How to handle labels more generally?
ham_with_corrections(p, t, α=1) = _ham_with_corrections(p..., t, α)
function _ham_with_corrections(ramp, ϵs, ζs, corrmax::Number, corrfull, P, t, α=1)
    Δs = ramp(t)
    Ham = (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
           ϵs[1] * P[0, 1] + ϵs[2] * P[2, 4] + ϵs[3] * P[3, 5] +
           -Δs[2] * ζs[1] * ζs[2] * P[1, 4] - Δs[3] * ζs[1] * ζs[3] * P[1, 5]) + corrmax * max_energy_correction_term(Δs, ζs, P)
    !iszero(corrfull) && (Ham += corrfull * full_energy_correction_term(Ham))
    return Ham * α
end
function _ham_with_corrections(ramp, ϵs, ζs, corrmax, corrfull, P, t, α=1)
    Δs = ramp(t)
    Ham = (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
           ϵs[1] * P[0, 1] + ϵs[2] * P[2, 4] + ϵs[3] * P[3, 5] +
           -Δs[2] * ζs[1] * ζs[2] * P[1, 4] - Δs[3] * ζs[1] * ζs[3] * P[1, 5]) + corrmax(t) * max_energy_correction_term(Δs, ζs, P)
    !iszero(corrfull) && (Ham += corrfull * full_energy_correction_term(Ham))
    return Ham * α
end

function ham_without_corrections((ramp, ϵs, ζs, P), t, α=1)
    Δs = ramp(t)
    Ham = (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
           ϵs[1] * P[0, 1] + ϵs[2] * P[2, 4] + ϵs[3] * P[3, 5])
    return Ham * α
end

function max_energy_correction_term(Δs, ζs, P)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ31 = √(Δs[3]^2 + Δs[1]^2)
    Δ12 = √(Δs[1]^2 + Δs[2]^2)
    (-ζs[1] * ζs[3] * Δ23 * Δs[3] / Δ31) * P[2, 4] +
    (-ζs[1] * ζs[2] * Δ23 * Δs[2] / Δ12) * P[3, 5]
end

function max_energy_correction_term!(ham, Δs, Δ23, Δ31, Δ12, P)
    @. ham += (-ζs[1] * ζs[3] * Δ23 * Δs[3] / Δ31) * P[2, 4] +
              (-ζs[1] * ζs[2] * Δ23 * Δs[2] / Δ12) * P[3, 5]
end

function full_energy_correction_term(ham)
    vals, vecs = eigen(Hermitian(ham))
    δE = 1 * (vals[1] - vals[2]) / 2
    δv = (vecs[:, 1] * vecs[:, 1]' - vecs[:, 2] * vecs[:, 2]') +
         (vecs[:, 3] * vecs[:, 3]' - vecs[:, 4] * vecs[:, 4]')
    return -δE * δv
end

function ham_with_corrections!(Ham, (ramp, ϵs, ζs, corr, P), t, α=1)
    Δs = ramp(t)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ31 = √(Δs[3]^2 + Δs[1]^2)
    Δ12 = √(Δs[1]^2 + Δs[2]^2)
    @. Ham = α * (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
                  ϵs[1] * P[0, 1] + (ϵs[2] - corr * ζs[1] * ζs[3] * Δ23 * Δs[3] / Δ31) * P[2, 4] + (ϵs[3] - corr * ζs[1] * ζs[2] * Δ23 * Δs[2] / Δ12) * P[3, 5] +
                  -(Δs[2]) * ζs[1] * ζs[2] * P[1, 4] - (Δs[3]) * ζs[1] * ζs[3] * P[1, 5])
    return Ham
end
