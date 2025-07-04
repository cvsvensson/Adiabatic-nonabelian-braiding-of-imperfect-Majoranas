visualize_spectrum(d::Dict) = visualize_spectrum(ham_with_corrections, d[:p], d[:ts])
function visualize_spectrum(H, p, ts,)
    spectrum = stack([eigvals(H(p, t)) for t in ts])'
    plot(ts, mapslices(v -> v[2:end] .- v[1], spectrum, dims=2), xlabel="t/T", ls=[:solid :dash :dot], ylabel="Eᵢ-E₀", labels=[1, 2, 3]', yscale=:log10, ylims=(1e-16, 1e1), lw=2, frame=:box)
end

visualize_rhos(d::Dict) = visualize_rhos(d[:k], d[:ts])
function visualize_rhos(k, ts)
    ρs = stack([get_rhos(k, t) for t in ts])'
    plot(ts, ρs, label=["ρMM̃" "ρML" "ρMR"], ylabel="ρs", xlabel="t/T", ls=[:solid :dash :dot], lw=3, frame=:box)
end
function visualize_protocol(dict::Dict)
    energyplot = visualize_spectrum(dict)
    deltaplot = visualize_rhos(dict)
    componentplot = visualize_analytic_parameters(dict)
    xlabel!(energyplot, "")
    xlabel!(deltaplot, "")
    plot(energyplot, deltaplot, componentplot, layout=(3, 1), size=400 .* (1, 1.5), frame=:box)
end

expectation_value(m::AbstractMatrix, ψ) = dot(ψ, m, ψ)
measure_parities(sol, dict::Dict, args...; kwargs...) = measure_parities(sol, dict[:P], args...; kwargs...)
const default_parity_pairs = [(:L, :L̃), (:M, :M̃), (:R, :R̃)]
function measure_parities(sol, P, parities=default_parity_pairs)
    [real(expectation_value(P[p], sol)) for p in parities]
end

function visualize_parities(sol, dict, parities=default_parity_pairs; ts=sol.t)
    P = dict[:P]
    label = permutedims(map(p -> join(p, ""), parities))
    plot(ts, [real(expectation_value(P[p], sol(t))) for p in parities, t in ts]'; label, legend=true, xlabel="t / T", ylims=(-1, 1), lw=2, frame=:box)
end

visualize_analytic_parameters(d::Dict) = visualize_analytic_parameters(d[:η], d[:k], d[:ts], d[:totalparity])
function visualize_analytic_parameters(η, k, ts, totalparity; opt_kwargs...)
    component_array_over_time = stack(zero_energy_analytic_parameters(η, k, t, totalparity; opt_kwargs...)[[:μ, :α, :β, :ν]] for t in ts)'
    component_labels = ["μ" "α" "β" "ν"]
    plot(ts, component_array_over_time, label=component_labels, xlabel="t / T", ylabel="Component", lw=2, frame=:box)
    plot!(legend=:topright)
end