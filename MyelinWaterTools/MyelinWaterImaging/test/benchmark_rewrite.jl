using BenchmarkTools
using LinearAlgebra

# "Pseuodo-random" numbers
prandmat(n) = exp.(sin.(reshape(1:n^2, n, n)))
prandvec(n) = sin.(1:n)

####
#### Nonnegative least squares
####

# n = 50;
# C = randn(n,n);
# d = randn(n);

# @btime T2Dist.Rewrite.lsqnonneg($C, $d);
# @btime NonNegLeastSquares.nonneg_lsq($C, $d; alg=:nnls);  # NNLS
# @btime NonNegLeastSquares.nonneg_lsq($C, $d; alg=:fnnls); # Fast NNLS
# @btime NonNegLeastSquares.nonneg_lsq($C, $d; alg=:pivot); # Pivot Method
# @btime NonNegLeastSquares.nonneg_lsq($C, $d; alg=:pivot, variant=:cache); # Pivot Method (cache pseudoinverse up front)
# @btime NonNegLeastSquares.nonneg_lsq($C, $d; alg=:pivot, variant=:comb); # Pivot Method with combinatorial least-squares

####
#### lsqnonneg_reg
####

n = 100;
C = randn(n,n); #prandmat(n);
d = randn(n); #prandvec(n);
Chi2Factor = 1.02;
work = T2Dist.Rewrite.lsqnonneg_reg_work(C, d);
@benchmark T2Dist.Rewrite.lsqnonneg_reg!($work, $C, $d, $Chi2Factor)
@benchmark T2Dist.Classic.lsqnonneg_reg($C, $d, $Chi2Factor)

####
#### Relaxation matrix action
####
ETL, TE, T2, T1 = 32, 10e-3, 20e-3, 1000e-3;
num_states, TE, T2, T1 = 32, 10e-3, 20e-3, 1000e-3;
E2, E1 = exp(-TE/T2), exp(-TE/T1);
flip_angle, num_pulses, refcon = 165.0, 32, 175.0;
work = T2Dist.Rewrite.EPGdecaycurve_work(Float64, ETL);

@benchmark begin
    T_r = T2Dist.Classic.relaxmat($num_states, $TE, $T2, $T1); # Sparse relaxation matrix
    T_r * $(work.M)
end
@benchmark T2Dist.Rewrite.relaxmat_action!($work, $num_states, $E2, $E1)

workf32, E2f32, E1f32 = map(x->Float32.(x), work), Float32(E2), Float32(E1);
@benchmark begin
    T2Dist.Rewrite.relaxmat_action!($workf32, $num_states, $E2f32, $E1f32)
end

####
#### Flip angle matrix action
####
T2mat = T2Dist.Rewrite.element_flip_mat(flip_angle * (refcon/180));
@benchmark begin
    T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat);
end
workf32, T2matf32 = map(x->Float32.(x), work), ComplexF32.(T2mat);
@benchmark begin
    T2Dist.Rewrite.flipmat_action!($workf32, $num_states, $T2matf32);
end

_, T_p = T2Dist.Classic.flipmat(deg2rad(flip_angle), num_pulses, refcon); # Sparse flip matrix
M_tmp = similar(work.M);
@benchmark begin
    mul!($M_tmp, $T_p, $(work.M))
end

T2mat = T2Dist.Rewrite.element_flip_mat(flip_angle * (refcon/180));
T2mat_elems = (real(T2mat[1,1]), real(T2mat[2,1]), -imag(T2mat[3,1]), -imag(T2mat[1,3]), real(T2mat[3,3]));
work = (M = randn(ComplexF64, 3*num_states), Mr = randn(3*num_states), Mi = randn(3*num_states));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat);

####
#### EPGdecaycurve
####
workf32, argsf32 = map(x->x isa Complex || eltype(x) <: Complex ? ComplexF32.(x) : Float32.(x), work), Float32.((flip_angle, TE, T2, T1, refcon))
@benchmark T2Dist.Rewrite.EPGdecaycurve!($work, $ETL, $flip_angle, $TE, $T2, $T1, $refcon)
@benchmark T2Dist.Rewrite.EPGdecaycurve!($workf32, $ETL, $argsf32...)

# @code_warntype T2Dist.Rewrite.EPGdecaycurve!(work, ETL, flip_angle, TE, T2, T1, refcon);
# @code_warntype T2Dist.Rewrite.EPGdecaycurve!(workf32, ETL, argsf32...);
