function single_qubit_gates(zero, one)
    @assert norm(zero) ≈ norm(one) ≈ 1 "basis states are not normalized"
    @assert abs(dot(zero, one)) < 1e-14 "basis states are not orthogonal"
    gates = Dict(:X => one * zero' + zero * one',
        :Y => -im * one * zero' + im * zero * one',
        :Z => zero * zero' - one * one',
        :I => zero * zero' + one * one')
    gates[:H] = sqrt(1 / 2) * (gates[:X] + gates[:Z])
    return gates
end

@testitem "Single gates" begin
    using LinearAlgebra
    basis = svd(rand(10, 2)).U
    v0, v1 = eachcol(basis) #random orthonormal states
    gates = MajoranaBraiding.single_qubit_gates(v0, v1)
    @test gates[:X] * v0 ≈ v1
    @test gates[:Y] * v0 ≈ -im * v1
    @test gates[:Z] * v0 ≈ v0
    @test gates[:Z] * v1 ≈ -v1
    @test gates[:H] * v0 ≈ (v0 + v1) / sqrt(2)
    @test basis' * gates[:X]' * gates[:X] * basis ≈ I

    @test gate_fidelity(gates[:X], gates[:X]) ≈ 1
    @test gate_fidelity(gates[:X], gates[:Y]) < 1e-14

    @test all(MajoranaBraiding.gate_overlaps(gates[S], gates)[S] ≈ 1 for S in (:X, :Y, :Z, :I))
end

function majorana_exchange(γ1, γ2)
    return sqrt(1im / 2) * (I + γ1 * γ2)
end
function majorana_exchange(parity_operator)
    return sqrt(1im / 2) * (I - 1im * parity_operator)
end


majorana_braid(γ1, γ2) = majorana_exchange(γ1, γ2)^2
majorana_braid(parity_operator) = majorana_exchange(parity_operator)^2

function gate_overlaps(gate, gates::Dict)
    Dict(k => tr(gate * v) / 2 for (k, v) in pairs(gates))
end

gate_fidelity(g1, g2) = abs(dot(g1, g2)^2 / (dot(g1, g1) * dot(g2, g2)))

@testitem "majorana_exchange" begin
    using LinearAlgebra, QuantumDots
    c = FermionBasis(1:2, qn=QuantumDots.parity)
    majorana_labels = 0:3
    γ = MajoranaWrapper(c, majorana_labels)

    vacuum_state_ind = c.symmetry.focktoinddict[0]
    vacuum_state = zeros(2^length(keys(c)))
    vacuum_state[vacuum_state_ind] = 1
    one_state = c[2]' * c[1]' * vacuum_state
    gates = single_qubit_gates(vacuum_state, one_state)
    basis = hcat(vacuum_state, one_state)

    B01 = majorana_exchange(γ[0], γ[1])
    @test B01' * B01 ≈ I
    @test B01 ≈ majorana_exchange(1im * γ[0] * γ[1])
    @test majorana_braid(γ[0], γ[1]) ≈ majorana_braid(1im * γ[0] * γ[1])

    @test majorana_exchange(γ[0], γ[1])^2 ≈ majorana_braid(γ[0], γ[1])

    majorana_braid(γ[1], γ[2])' * majorana_braid(γ[1], γ[2]) ≈ I
    @test gate_overlaps(majorana_braid(γ[1], γ[2]), gates)[:X] ≈ -1


    @test gate_overlaps(majorana_braid(γ[1], γ[3]), gates)[:Y] ≈ -1
end