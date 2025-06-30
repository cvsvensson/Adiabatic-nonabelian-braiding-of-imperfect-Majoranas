module MajoranaBraiding
using LinearAlgebra
using StaticArrays
using Optim, Interpolations
using TestItems
using Roots
using Plots
using UnPack
using OrdinaryDiffEqCore
using FermionicHilbertSpaces

export RampProtocol
export parity_operators
export majorana_exchange, gate_fidelity, analytical_gate_fidelity
export SimpleCorrection, IndependentSimpleCorrection, InterpolatedExactSimpleCorrection, NoCorrection
export single_braid_gate_kato, single_braid_gate_analytical_angle, single_braid_gate_analytical
export setup_problem
export visualize_protocol, visualize_parities, visualize_analytic_parameters, visualize_spectrum, visualize_deltas
export diagonal_majoranas, OptimizedSimpleCorrection, IndependentSimpleCorrection, OptimizedIndependentSimpleCorrection
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
