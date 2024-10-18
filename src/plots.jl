
function visualize_protocol(H, ramp, ϵs, ζs, corr, P, ts)
    p = (ramp, ϵs, ζs, corr, P)
    deltas = stack([ramp(t) for t in ts])'
    delta_plot = plot(ts, deltas, label=["Δ01" "Δ02" "Δ03"], xlabel="t", ls=[:solid :dash :dot], lw=3)
    spectrum = stack([eigvals(H(p, t)) for t in ts])'
    plot(plot(ts, mapslices(v -> v[2:end] .- v[1], spectrum, dims=2), ls=[:solid :dash :dot], title="Eᵢ-E₀", labels=[1, 2, 3]', yscale=:log10, ylims=(1e-16, 1e1)), delta_plot, layout=(2, 1), lw=2, frame=:box)
end
visualize_protocol(dict::Dict) = visualize_protocol(ham_with_corrections, dict[:ramp], dict[:ϵs], dict[:ζ], dict[:correction], dict[:P], dict[:ts])

expval(m::AbstractMatrix, ψ) = dot(ψ, m, ψ)
measure_parities(sol, dict::Dict, args...; kwargs...) = measure_parities(sol, dict[:P], args...; kwargs...)
function measure_parities(sol, P, parities=[(0, 1), (2, 4), (3, 5)])
    [real(expval(P[p...], sol)) for p in parities]
end

function visualize_parities(sol, P, parities=[(0, 1), (2, 4), (3, 5)]; ts=sol.t)
    ts = sol.t
    measurements = map(p -> P[p...], parities)
    plot(ts, [real(expval(m, sol(t))) for m in measurements, t in ts]', label=permutedims(parities), legend=true, xlabel="t", ylims=(-1, 1), lw=2, frame=:box)
end
visualize_parities(sol, dict::Dict, parities=[(0, 1), (2, 4), (3, 5)]) = visualize_parities(sol, dict[:P], parities)