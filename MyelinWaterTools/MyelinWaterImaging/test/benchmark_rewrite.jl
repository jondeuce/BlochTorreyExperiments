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
work = T2Dist.Rewrite.lsqnonneg_reg_work(C, d, Chi2Factor);
@benchmark T2Dist.Rewrite.lsqnonneg_reg!($work, $C, $d, $Chi2Factor)
@benchmark T2Dist.Classic.lsqnonneg_reg($C, $d, $Chi2Factor)

####
#### relaxmat, flipmat, and EPGdecaycurve
####
ETL, TE, T2, T1 = 32, 10e-3, 20e-3, 1000e-3;
num_states, TE, T2, T1 = 32, 10e-3, 20e-3, 1000e-3;
E2, E1 = exp(-TE/T2), exp(-TE/T1);
flip_angle, num_pulses, refcon = 165.0, 32, 175.0;
work = T2Dist.Rewrite.EPGdecaycurve_work(ETL);

@benchmark begin
    T_r = T2Dist.Classic.relaxmat($num_states, $TE, $T2, $T1); # Sparse relaxation matrix
    T_r * $(work.M)
end

all_res = let work = (M = randn(ComplexF64, 3*num_states), M_tmp = randn(ComplexF64, 3*num_states))
    [T2Dist.Rewrite.relaxmat_action!(work, num_states, E2, E1, Val(N)) for N in 1:2]
end;
@assert all([all_res[i] ≈ all_res[i+1] for i in 1:length(all_res)-1])

@benchmark T2Dist.Rewrite.relaxmat_action!($work, $num_states, $TE, $T2, $T1, Val(1))
@btime T2Dist.Rewrite.relaxmat_action!($work, $num_states, $E2, $E1, Val(1));
@btime T2Dist.Rewrite.relaxmat_action!($work, $num_states, $E2, $E1, Val(1));
@btime T2Dist.Rewrite.relaxmat_action!($work, $num_states, $E2, $E1, Val(2));
@btime T2Dist.Rewrite.relaxmat_action!($work, $num_states, $E2, $E1, Val(2));

####
#### EPGdecaycurve
####
T2mat = T2Dist.Rewrite.element_flip_mat(flip_angle * (refcon/180));
@benchmark begin
    T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat);
end
_, T_p = T2Dist.Classic.flipmat(deg2rad(flip_angle), num_pulses, refcon); # Sparse flip matrix
@benchmark begin
    mul!($(work.M_tmp), $T_p, $(work.M))
end

T2mat = T2Dist.Rewrite.element_flip_mat(flip_angle * (refcon/180));
all_res = let work = (M = randn(ComplexF64, 3*num_states),)
    [T2Dist.Rewrite.flipmat_action!(work, num_states, T2mat, Val(N)) for N in 1:8]
end;
@assert all([all_res[i] ≈ all_res[i+1] for i in 1:length(all_res)-1])

T2mat_elems = (real(T2mat[1,1]), real(T2mat[2,1]), -imag(T2mat[3,1]), -imag(T2mat[1,3]), real(T2mat[3,3]))
work = (M = randn(ComplexF64, 3*num_states), Mr = randn(3*num_states), Mi = randn(3*num_states))
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(1));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(2));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(3));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(4));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(5));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(6));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(7));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(8));
@btime T2Dist.Rewrite.flipmat_action!($work, $num_states, $T2mat, Val(9));

####
#### EPGdecaycurve
####
@benchmark T2Dist.Rewrite.EPGdecaycurve!($work, $ETL, $flip_angle, $TE, $T2, $T1, $refcon)
