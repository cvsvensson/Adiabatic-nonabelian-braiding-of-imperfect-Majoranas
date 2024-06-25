
#How to handle labels more generally?
function ham_with_corrections((ramp, ϵs, ζs, corr, P), t, α=1)
    Δs = ramp(t)
    Δ23 = √(Δs[2]^2 + Δs[3]^2)
    Δ31 = √(Δs[3]^2 + Δs[1]^2)
    Δ12 = √(Δs[1]^2 + Δs[2]^2)
    Ham = α * (Δs[1] * P[0, 1] + Δs[2] * P[0, 2] + Δs[3] * P[0, 3] +
               ϵs[1] * P[0, 1] + (ϵs[2] - corr * ζs[1] * ζs[3] * Δ23 * Δs[3] / Δ31) * P[2, 4] + (ϵs[3] - corr * ζs[1] * ζs[2] * Δ23 * Δs[2] / Δ12) * P[3, 5] +
               -(Δs[2]) * ζs[1] * ζs[2] * P[1, 4] - (Δs[3]) * ζs[1] * ζs[3] * P[1, 5])
    return Ham
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
