module MajoranaBraiding
using LinearAlgebra
using OrdinaryDiffEq
using StaticArrays
using QuantumDots
using Optim, Interpolations
using TestItems
using Roots
using Majoranas

export MajoranaWrapper
export RampProtocol
export parity_operators
export single_qubit_gates, gate_overlaps, majorana_exchange, majorana_braid, gate_fidelity
export ham_with_corrections, ham_with_corrections!, get_op

include("majoranas.jl")
include("misc.jl")
include("hamiltonians.jl")
include("protocol.jl")
include("gates.jl")

@static if false
    include("../scripts/braiding_extra_majoranas.jl")
end

end
