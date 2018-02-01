using NSGAII
using Compat.Test, Compat.Random


const C1 = [2 5 4 7 ; 3 3 5 7 ; 3 8 4 2 ; 6 5 2 5]
const C2 = [3 3 6 2 ; 5 3 7 3 ; 5 2 7 4 ; 4 6 3 5]
const C3 = [4 2 5 3 ; 5 3 4 3 ; 4 3 5 2 ; 6 4 7 3]

z(x, C) = sum(inds->C[inds...], enumerate(x))
z(x::Vector{Int}) = z(x, C1), z(x, C2), z(x, C3)

res = unique(nsga(500, 100, z, ()->randperm(4)))
@test length(res) == 7


const d = RealCoding(6, [-10], [10])
z(x) = x[1]^2, (x[1] - 2)^2
seed = [-10 + rand()*20 for _ =1:100]
res = nsga(500, 200, z, d, seed = seed)

@test maximum(x -> x.y[1], res) >= 3.99
@test minimum(x -> x.y[1], res) <= 1e-4
@test maximum(x -> x.y[2], res) >= 3.99
@test minimum(x -> x.y[2], res) <= 1e-4
