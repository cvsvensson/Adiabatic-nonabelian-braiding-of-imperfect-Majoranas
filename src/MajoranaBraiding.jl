module MajoranaBraiding
using LinearAlgebra
using OrdinaryDiffEq
using StaticArrays
using QuantumDots
using Optim, Interpolations
using TestItems
using Roots
using Majoranas
using Plots

export MajoranaWrapper
export RampProtocol
export parity_operators
export single_qubit_gates, gate_overlaps, majorana_exchange, majorana_braid, gate_fidelity
export SimpleCorrection, IndependentSimpleCorrections, EigenEnergyCorrection, WeakEnergyCorrection, InterpolatedExactSimpleCorrection, NoCorrection
export optimized_simple_correction, optimized_independent_simple_correction, analytical_exact_simple_correction, find_zero_energy_from_analytics, single_braid_gate_kato, braid_gate_best_angle, braid_gate_prediction, single_braid_gate_analytical_angles
export ham_with_corrections, get_op
export visualize_protocol, visualize_parities, visualize_groundstate_components, visualize_spectrum, visualize_deltas
export diagonal_majoranas
export measure_parities

include("majoranas.jl")
include("misc.jl")
include("hamiltonians.jl")
include("analytic_correction.jl")
include("protocol.jl")
include("gates.jl")
include("plots.jl")

@static if false
    include("../scripts/braiding_extra_majoranas.jl")
    include("../scripts/eigen_correction_dec.jl")
end

end
