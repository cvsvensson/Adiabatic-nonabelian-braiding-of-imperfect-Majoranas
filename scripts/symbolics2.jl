
using MajoranaBraiding
using QuantumDots
using Symbolics
# using SymPy
# using Nemo
struct MajoranaWrapper{B,D}
    basis::B
    majoranas::D
    function MajoranaWrapper(basis, labels=collect(Base.product(keys(basis), (:a, :b))))
        N = length(basis)
        length(labels) == 2 * N || throw(ErrorException("Number of majoranas is not twice the fermion number"))
        majA = map(f -> f + f', basis)
        majB = map(f -> 1im * (f - f'), basis)
        majs = vcat(majA, majB)
        dA = QuantumDots.dictionary(zip(labels[1:N], values(majA)))
        dB = QuantumDots.dictionary(zip(labels[N+1:2N], values(majB)))
        d = merge(dA, dB)
        new{typeof(basis),typeof(d)}(basis, d)
    end
end
Base.getindex(g::MajoranaWrapper, i...) = g.majoranas[i...]
##
γmb = get_majorana_basis()
fbdg = FermionBdGBasis(1:3)
γbdg = MajoranaWrapper(fbdg, collect(keys(γmb)))
N = length(γmb.fermion_basis)
@majoranas γ
mtype, vtype = MajoranaBraiding.matrix_vec_types(true, false, N)
P = parity_operators(γmb, 1, mtype)
get_P(γ) = Dict((l1, l2) => (1im * γ[l1] * γ[l2]) for l1 in keys(γmb), l2 in keys(γmb))
get_iP(γ) = Dict((l1, l2) => (-γ[l1] * γ[l2]) for l1 in keys(γmb), l2 in keys(γmb))
Psym = get_P(γ)
Pbdg = get_P(γbdg)

## Middle of the protocol
SymPy.@syms Δ::real, λ::real, ζ::real
get_ham(P; Δ=Δ, λ=λ, ζ=ζ) = (Δ * P[:M, :L] + Δ * P[:M, :R] + λ * (P[:L, :L̃] + P[:R, :R̃]) + Δ * ζ * ζ * P[:M̃, :L̃] + Δ * ζ * ζ * P[:M̃, :R̃])
symhammb = get_ham(P)
symhamsym = get_ham(Psym)
symhambdg = get_ham(Pbdg)
vecs, vals = SymPy.Matrix(symhammb).diagonalize()

##
@variables Δ::Real λ::Real ζ::Real λ2::Real
energies = [-λ - 2 * (1 // 2 * Δ^2 * ζ^4 - Δ^2 * ζ^2 + 1 // 2 * Δ^2 + 1 // 4 * λ^2)^(1 // 2)
    -λ + 2 * (1 // 2 * Δ^2 * ζ^4 - Δ^2 * ζ^2 + 1 // 2 * Δ^2 + 1 // 4 * λ^2)^(1 // 2)
    λ - 2 * (1 // 2 * Δ^2 * ζ^4 + Δ^2 * ζ^2 + 1 // 2 * Δ^2 + 1 // 4 * λ^2)^(1 // 2)
    λ + 2 * (1 // 2 * Δ^2 * ζ^4 + Δ^2 * ζ^2 + 1 // 2 * Δ^2 + 1 // 4 * λ^2)^(1 // 2)]
eq = energies[1] ~ energies[3]
# first and third energies are the smallest ones for λ ∼ 0 and ζ ∼ 0

symbolic_solve(eq, λ)
symsol = symbolic_solve((energies[1] - energies[2])^2 ~ 0, λ)
symbolic_solve(substitute((energies[1] - energies[2])^2, λ => λ2^0.5) / 16 ~ 0, λ2)
symbolic_solve(eq, λ)

##
Symbolics.simplify.(substitute.(get_ham(P), λ => symsol[1]))
Symbolics.simplify.(substitute.(energies, λ => symsol[1]))

##
labels = collect(keys(γmb))
ind_dict = Dict(zip(labels, eachindex(labels)))
@variables _x[1:6] _y[1:6]
x = Dict(zip(labels, _x))
y = Dict(zip(labels, _y))
γL = sum(x[ind_dict[l]] * γ[l] for l in labels)
γR = sum(y[ind_dict[l]] * γ[l] for l in labels)
##
relations = [y[1] => x[1] * y[2] / x[2],
    y[3] => x[3] * y[6] / x[6],
    y[4] => x[4] * y[5] / x[5],
    y[6] => (x[1] * x[6] * y[2] / x[2]) / x[1]]
display.(map((kv) -> kv[1] => substitute(kv[2], relations), collect(pairs((γL * γR).dict))))

# x[3]*y[6] - x[6]*y[3] ==0 => y[3] => x[3]*y[6]/x[6]
# -x[4]*y[5] + x[5]*y[4]==0 => y[4] => x[4]*y[5]/x[5]
#(-x[1]*x[6]*y[2]) / x[2] + x[1]*y[6]==0 => y[6] => (x[1]*x[6]*y[2]/x[2])/x[1]
#(x[3]*x[4]*y[5]) / x[5] + (-x[3]*x[4]*y[6]) / x[6]==0 
##
eqs = collect(values((γL * γR - get_ham(get_iP(γ))).dict))
symmetry_relations = Dict(y[1] => x[1], y[4] => x[3], y[3] => x[4], y[6] => x[5], y[5] => x[6])
simplified_eqs = map(eq -> substitute(eq, symmetry_relations), eqs)
eq_relations = []
new_eqs = foldl((eqs, subs) -> map(eq -> substitute(eq, subs), eqs), eq_relations; init=simplified_eqs)
display.(new_eqs)
##
eqs = collect(values((γL * γR - get_ham(get_iP(γ))).dict))
symmetry_relations = Dict()
simplified_eqs = map(eq -> simplify(substitute(eq, symmetry_relations)), eqs)
eq_relations = [x[5] => 0,];
new_eqs = foldl((eqs, subs) -> map(eq -> simplify(substitute(eq, subs)), eqs), eq_relations; init=simplified_eqs);
display.(new_eqs);

## let's solve the ζ = 0 case first
eqs = collect(values((γL * γR - get_ham(get_iP(γ))).dict))
symmetry_relations = Dict(ζ => 0, λ => 0, y[2] => 0, x[2] => 0, y[5] => 0, y[6] => 0, x[5] => 0, x[6] => 0)
simplified_eqs = map(eq -> substitute(eq, symmetry_relations), eqs)
eq_relations = [x[4] => x[3] * y[4] / y[3], x[3] => Δ / y[1] + x[1] * y[3] / y[1], y[4] => y[3]]#, y[1] => (1 - 2y[3]^2)^0.5, y[3] => 0]
new_eqs = foldl((eqs, subs) -> map(eq -> simplify(substitute(eq, subs)), eqs), eq_relations; init=simplified_eqs)
display.(new_eqs)

## Apply these relations to γL
Ldict = sort(γL.dict, by=x -> first(x.factors))
Rdict = sort(γR.dict, by=x -> first(x.factors))
foldl((eqs, subs) -> map(eq -> simplify(substitute(eq, subs)), eqs), vcat(symmetry_relations, eq_relations); init=collect(values(Ldict)))
foldl((eqs, subs) -> map(eq -> simplify(substitute(eq, subs)), eqs), vcat(symmetry_relations, eq_relations); init=collect(values(Rdict)))
foldl((eqs, subs) -> map(eq -> simplify(substitute(eq, subs)), eqs), vcat(symmetry_relations, eq_relations); init=collect(values(γL^2)))
foldl((eqs, subs) -> map(eq -> simplify(substitute(eq, subs)), eqs), vcat(symmetry_relations, eq_relations); init=collect(values(γR^2))) |> simplify

#symbolic_solve(collect(values((γL * γR - get_ham(get_iP(γ))).dict)), y[1:6])
##

map(eq -> simplify(substitute(eq, [Δ => 1, ζ => 0.1])), simplified_eqs)
##
using Optim
using LinearAlgebra
function cost_function(xs; Δ=1, ζ=0.2, P=P, γ=γmb)
    x = xs[1:6]
    y = xs[7:12]
    λ = xs[13]
    c0 = xs[14]
    α = xs[15]
    labels = sort(collect(keys(γmb)))
    ind_dict = Dict(zip(labels, eachindex(labels)))
    ham = 1im * (Δ * P[:M, :L] + Δ * P[:M, :R] + λ * (P[:L, :L̃] + P[:R, :R̃]) + Δ * ζ * ζ * P[:M̃, :L̃] + Δ * ζ * ζ * P[:M̃, :R̃] + c0 * I)
    γL = sum(x[ind_dict[l]] * γmb[l] for l in labels)
    γR = sum(y[ind_dict[l]] * γmb[l] for l in labels)
    # eqs = collect(values((γL * γR - ham).dict))
    norm(α * (γL*γR)[5:8, 5:8] - ham) + (norm(x) - 1)^2 + (norm(y) - 1)^2
end
cost_function(rand(15))
#optimize with optim
res = optimize(cost_function, rand(15), Newton())
##


#why is this sqrt(2)*2/100???? (it's 2sqrt(2)Δ*ζ^2)

##

resλ = optimize(λ -> abs(eigvals(Matrix(get_ham(Pbdg; Δ=1, ζ=0.1, λ=λ[1])))[2]), [0.0], NelderMead())


resλ = optimize(λ -> abs(diff(eigvals(Matrix(get_ham(P; Δ=1, ζ=0.1, λ=λ[1]))))[1]), [0.0], NelderMead())

##
xsyms = Tuple(Symbol(:x,l) for l in labels)
ysyms = Tuple(Symbol(:y,l) for l in labels)
x = Dict(zip(labels,[only(@variables $s) for s in xsyms]))
y = Dict(zip(labels,[only(@variables $s) for s in ysyms]))
γL = sum(x[l] * γ[l] for l in labels)
γR = sum(y[l] * γ[l] for l in labels)
let H = get_ham(get_iP(γ))
    #(1im*(H*γL*γR - γL*γR*H)).dict
    eqs = collect(values((H * γL * γR - γL * γR * H).dict))
    symmetry_relations = Dict(y[:L] => y[:R], y[:L̃] => 0, y[:R̃] => 0, x[:L] => 0, x[:R] => 0, x[:L̃] => x[:R̃])
    simplified_eqs = map(eq -> substitute(eq, symmetry_relations), eqs)
end