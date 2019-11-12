using Test, BenchmarkTools
using Parameters: @unpack
using MATLAB
using T2Dist

# "Pseuodo-random" numbers
prandmat(n) = exp.(sin.(reshape(1:n^2, n, n)))
prandvec(n) = sin.(1:n)

####
#### lsqnonneg_reg
####

n = 100;
args = (C = prandmat(n), d = prandvec(n), Chi2Factor = 1.02)
# @unpack C, d, Chi2Factor = args
jl = T2Dist.Classic.lsqnonneg_reg(args...)
mat = mxcall(:lsqnonneg_reg, 3, args...)
@assert all(map(isapprox, jl, mat))

####
#### EPGdecaycurve
####

args = (ETL = 32, flip_angle = 50.0, TE = 10e-3, T2 = 15e-3, T1 = 1000e-3, refcon = 180.0)
# @unpack ETL, flip_angle, TE, T2, T1, refcon = args
jl = T2Dist.Classic.EPGdecaycurve(args...)
mat = mxcall(:EPGdecaycurve, 1, Float64.(values(args))...)
@assert all(map(isapprox, jl, mat))

####
#### T2map_SEcorr
####

n = 10
M = 1e4 .* reshape(exp.(.-(1/6.0).*(1:32)) .+ exp.(.-(1/2.5).*(1:32)), 1, 1, 1, :);
image = repeat(M, n, n, n, 1);
@time jl = T2Dist.Classic.T2map_SEcorr(image);
mat = mxcall(:T2map_SEcorr, 2, image, "waitbar", "no");

for k in keys(jl[1])
    s = string(k)
    if haskey(mat[1], s)
        @assert jl[1][k] ≈ mat[1][s]
    end
end
@assert jl[2] ≈ mat[2]

nothing