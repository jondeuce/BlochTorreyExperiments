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
C = prandmat(n);
d = prandvec(n);
Chi2Factor = 1.02;
work = T2Dist.Rewrite.lsqnonneg_reg_work(C, d, Chi2Factor);
@benchmark T2Dist.Rewrite.lsqnonneg_reg!($work, $C, $d, $Chi2Factor)
@benchmark T2Dist.Classic.lsqnonneg_reg($C, $d, $Chi2Factor)

####
#### relaxmat, flipmat, and EPGdecaycurve
####
ETL, TE, T2, T1 = 32, 10e-3, 20e-3, 1000e-3;
num_states, te, t2, t1 = 32, 10e-3, 20e-3, 1000e-3;
flip_angle, num_pulses, refcon = 165.0, 32, 175.0;
work = T2Dist.Rewrite.EPGdecaycurve_work(ETL, flip_angle, TE, T2, T1, refcon);

@benchmark T2Dist.Rewrite.relaxmat_action!($work, $num_states, $te, $t2, $t1)
@benchmark begin
    T_r = T2Dist.Classic.relaxmat($num_states, $te, $t2, $t1); # Sparse relaxation matrix
    T_r * $(work.M)
end

@benchmark begin
    T1mat, T2mat = T2Dist.Rewrite.flip_matrices($flip_angle, $num_pulses, $refcon);
    T2Dist.Rewrite.flipmat_action!($work, $num_states, T2mat);
end
@benchmark begin
    _, T_p = T2Dist.Classic.flipmat(deg2rad($flip_angle), $num_pulses, $refcon); # Sparse flip matrix
    T_p * $(work.M)
end

@benchmark T2Dist.Rewrite.EPGdecaycurve!($work, $ETL, $flip_angle, $TE, $T2, $T1, $refcon)
