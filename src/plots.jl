visualize_spectrum(d::Dict) = visualize_spectrum(ham_with_corrections, d[:p], d[:ts], d[:T])
function visualize_spectrum(H, p, ts, T)
    spectrum = stack([eigvals(H(p, t)) for t in ts])'
    plot(ts / T, mapslices(v -> v[2:end] .- v[1], spectrum, dims=2), xlabel="t/T", ls=[:solid :dash :dot], ylabel="Eᵢ-E₀", labels=[1, 2, 3]', yscale=:log10, ylims=(1e-16, 1e1), lw=2, frame=:box)
end

visualize_deltas(d::Dict) = visualize_deltas(d[:ramp], d[:ts], d[:T])
function visualize_deltas(ramp, ts, T)
    deltas = stack([ramp(t) for t in ts])'
    plot(ts / T, deltas, label=["ΔMM̃" "ΔML" "ΔMR"], ylabel="Δs", xlabel="t/T", ls=[:solid :dash :dot], lw=3, frame=:box)
end
function visualize_protocol(dict::Dict)
    energyplot = visualize_spectrum(dict)
    deltaplot = visualize_deltas(dict)
    componentplot = visualize_analytic_parameters(dict)
    xlabel!(energyplot, "")
    xlabel!(deltaplot, "")
    plot(energyplot, deltaplot, componentplot, layout=(3, 1), size=400 .* (1, 1.5), frame=:box)
end

expval(m::AbstractMatrix, ψ) = dot(ψ, m, ψ)
measure_parities(sol, dict::Dict, args...; kwargs...) = measure_parities(sol, dict[:P], args...; kwargs...)
default_parity_pairs = [(:L, :L̃), (:M, :M̃), (:R, :R̃)]
function measure_parities(sol, P, parities=default_parity_pairs)
    [real(expval(P[p...], sol)) for p in parities]
end

parity_labels(parities) = permutedims(map(p -> join(p, ""), parities))
function visualize_parities(sol, P, T, parities=default_parity_pairs; ts=sol.t)
    measurements = map(p -> P[p...], parities)
    plot(ts / T, [real(expval(m, sol(t))) for m in measurements, t in ts]', label=parity_labels(parities), legend=true, xlabel="t / T", ylims=(-1, 1), lw=2, frame=:box)
end
visualize_parities(sol, dict::Dict, parities=default_parity_pairs) = visualize_parities(sol, dict[:P], dict[:T], parities)

visualize_analytic_parameters(d::Dict) = visualize_analytic_parameters(d[:ζ], d[:ramp], d[:ts], d[:T], get(d, :totalparity, 1))
function visualize_analytic_parameters(ζ, ramp, ts, T, totalparity; opt_kwargs...)
    component_array_over_time = stack(zero_energy_analytic_parameters(ζ, ramp, t, totalparity; opt_kwargs...)[["μ" "α" "β" "ν"]] for t in ts)'
    component_labels = ["μ" "α" "β" "ν"]
    plot(ts / T, component_array_over_time, label=component_labels, xlabel="t / T", ylabel="Component", lw=2, frame=:box)
    plot!(legend=:topright)
end