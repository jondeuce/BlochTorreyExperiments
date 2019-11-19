using Test, BenchmarkTools
using Parameters: @unpack
using MATLAB, MAT
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
num_states, TE, T2, T1 = 32, 10e-3, 20e-3, 1000e-3;
flip_angle, num_pulses, refcon = 165.0, 32, 175.0;
E2, E1 = exp(-TE/T2), exp(-TE/T1);

M = randn(ComplexF64, 3*num_states);
work = (M=copy(M),);
T2Dist.Rewrite.relaxmat_action!(work.M, num_states, E2, E1);
T_r = T2Dist.Classic.relaxmat(num_states, TE, T2, T1); # Compare with classic T_r sparse relaxation matrix
@assert T_r * M ≈ work.M

work = (M=copy(M),);
T2mat = T2Dist.Rewrite.element_flip_mat(flip_angle * (refcon/180));
T2Dist.Rewrite.flipmat_action!(work.M, num_states, T2mat);
_, T_p = T2Dist.Classic.flipmat(deg2rad(flip_angle), num_pulses, refcon); # Compare with classic T_p sparse flip matrix
@assert T_p * M ≈ work.M

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
image = repeat(M, n, n, n, 1); # image = Float32.(image);
time_jl = @elapsed(jl = T2Dist.Rewrite.T2mapSEcorr(image)); @show time_jl; @show time_jl * 1e6/n^3;
mat = mxcall(:T2map_SEcorr, 2, image, "waitbar", "no");

for s in keys(jl[1])
    haskey(mat[1], s) && @assert isapprox(jl[1][s], mat[1][s]; rtol = 1e-4)
end
@assert isapprox(jl[2], mat[2]; rtol = 1e-4)

# T2map_SEpart
m = 64;
t2d = jl[2][1,1,1,:]; #mat[2][1,1,1,:];
t2dist = repeat(reshape(t2d,1,1,1,:), m, m, m, 1);
@time jlp = T2Dist.Rewrite.T2partSEcorr(t2dist; Sigmoid = 0.2);
# @btime T2Dist.Rewrite.T2partSEcorr($t2dist);
matp = mxcall(:T2part_SEcorr, 1, t2dist, "Sigmoid", 0.2);
for s in keys(jlp)
    haskey(matp, s) && @assert isapprox(jlp[s], matp[s])
end

####
#### Real MWI example
####
base_folder = "/home/jdoucette/Documents/code/MWIProcessing/Example_48echo_8msTE/"
base_filename = "ORIENTATION_B0_08_WIP_MWF_CPMG_CS_AXIAL_5_1"
data = MAT.matread(joinpath(base_folder, base_filename * ".mat"));
maps, dist = T2Dist.Rewrite.T2mapSEcorr(
    data["img"];
    TE = 8e-3,
);
MAT.matwrite(joinpath(base_folder, "julia/", base_filename * ".t2maps.jl.mat"), maps);
MAT.matwrite(joinpath(base_folder, "julia/", base_filename * ".t2dist.jl.mat"), Dict("dist" => dist));

dist = MAT.matread(joinpath(base_folder, "julia/", base_filename * ".t2dist.jl.mat"));
@time mwimaps = T2Dist.Rewrite.T2partSEcorr(dist["dist"]; SPWin = (14e-3, 40e-3))
MAT.matwrite(joinpath(base_folder, "julia/", base_filename * ".mwimaps-spwin_14e-3_40e-3.jl.mat"), mwimaps);

using StatsPlots
rand_ijk() = (i = rand(1:size(data["img"],1)), j = rand(1:size(data["img"],2)), k = rand(1:size(data["img"],3)))
rand_valid() = (ijk = rand_ijk(); while data["img"][ijk...,1] < 200; ijk = rand_ijk(); end; ijk)

plot(dist[rand_valid()...,:])

nothing