module MajoranaBraiding
using LinearAlgebra
using StaticArrays
using QuantumDots
using Optim, Interpolations
using TestItems
using Roots
using Majoranas
using Plots
using UnPack
using OrdinaryDiffEqCore

export MajoranaWrapper
export RampProtocol
export parity_operators, get_majorana_basis
export single_qubit_gates, gate_overlaps, majorana_exchange, majorana_braid, gate_fidelity, analytical_gate_fidelity
export SimpleCorrection, IndependentSimpleCorrections, EigenEnergyCorrection, WeakEnergyCorrection, InterpolatedExactSimpleCorrection, NoCorrection
export optimized_simple_correction, optimized_independent_simple_correction, analytical_exact_simple_correction, find_zero_energy_from_analytics, single_braid_gate_kato, braid_gate_best_angle, braid_gate_prediction, single_braid_gate_analytical_angles, single_braid_gate_analytical_angle, single_braid_gate_analytical
export ham_with_corrections, get_op, setup_problem
export visualize_protocol, visualize_parities, visualize_analytic_parameters, visualize_spectrum, visualize_deltas
export diagonal_majoranas, OptimizedSimpleCorrection
export measure_parities

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
