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
jl = T2Dist.Rewrite.lsqnonneg_reg(args...)
mat = mxcall(:lsqnonneg_reg, 3, args...)
@assert all(map(isapprox, jl, mat))

####
#### relaxmat and flipmat
####
num_states, te, t2, t1 = 32, 10e-3, 20e-3, 1000e-3;
flip_angle, num_pulses, refcon = 165.0, 32, 175.0;

M = randn(ComplexF64, 3*num_states);
M_tmp = randn(ComplexF64, 3*num_states);

work = (M=copy(M), M_tmp=copy(M_tmp));
T2Dist.Rewrite.relaxmat_action!(work, num_states, te, t2, t1);
T_r = T2Dist.Classic.relaxmat(num_states, te, t2, t1); # Compare with classic T_r sparse relaxation matrix
T_r * M ≈ work.M

work = (M=copy(M),);
T1mat, T2mat = T2Dist.Rewrite.flip_matrices(flip_angle, num_pulses, refcon);
T2Dist.Rewrite.flipmat_action!(work, num_states, T2mat);
_, T_p = T2Dist.Classic.flipmat(deg2rad(flip_angle), num_pulses, refcon); # Compare with classic T_p sparse flip matrix
T_p * M ≈ work.M

####
#### EPGdecaycurve
####

args = (ETL = 32, flip_angle = 50.0, TE = 10e-3, T2 = 15e-3, T1 = 1000e-3, refcon = 180.0);
# @unpack ETL, flip_angle, TE, T2, T1, refcon = args;
jl = T2Dist.Rewrite.EPGdecaycurve(args...);
mat = mxcall(:EPGdecaycurve, 1, Float64.(values(args))...);
@assert all(map(isapprox, jl, mat))

####
#### T2map_SEcorr
####

n = 10
M = 1e4 .* reshape(exp.(.-(1/6.0).*(1:32)) .+ exp.(.-(1/2.5).*(1:32)), 1, 1, 1, :);
image = repeat(M, n, n, n, 1);
@time jl = T2Dist.Rewrite.T2map_SEcorr(image);
mat = mxcall(:T2map_SEcorr, 2, image, "waitbar", "no");

for k in keys(jl[1])
    s = string(k)
    if haskey(mat[1], s)
        @assert isapprox(jl[1][k], mat[1][s]; rtol = 1e-4)
    end
end
@assert isapprox(jl[2], mat[2]; rtol = 1e-4)

nothing