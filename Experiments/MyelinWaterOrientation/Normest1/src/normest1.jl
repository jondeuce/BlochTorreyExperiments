# ============================================================================ #
# Semi-direct translation of the MATLAB `normest1` code
# ============================================================================ #

# @static if VERSION >= v"0.7.0"
#    A_mul_B!(y, A, x)  = mul!(y, A, x)
#    At_mul_B!(y, A, x) = mul!(y, transpose(A), x)
#    Ac_mul_B!(y, A, x) = mul!(y, adjoint(A), x)
# end

# @static if VERSION < v"0.7.0"
#    # Define multiplication of LinearMap on matrices (taken from v0.7 branch of LinearMaps.jl)
#    for f in (:A_mul_B!, :At_mul_B!, :Ac_mul_B!)
#       @eval function LinearAlgebra.$f(Y::AbstractMatrix, A::LinearMap{Te}, X::AbstractMatrix) where {Te}
#          @inbounds @views for i = 1:size(X, 2)
#             $f(Y[:, i], A, X[:, i])
#          end
#          return Y
#       end
#    end
# end

# Wrap `normest1` below to return the norm estimate only.
function normest1_norm(A, p::Real = 1, t::Int = 2)
   !(size(A,1) == size(A,2)) && error("Matrix A must be square")
   !(p == 1 || p == Inf) && error("Only p=1 or p=Inf supported")
   p == Inf && (A = A')
   t = min(t, size(A,2))
   return normest1(A, t)[1]
end

#NORMEST1 Estimate of 1-norm of matrix by block 1-norm power method.
#   C = NORMEST1(A) returns an estimate C of norm(A,1), where A is N-by-N.
#   A can be an explicit matrix or a function AFUN such that
#   FEVAL(@AFUN,FLAG,X) for the following values of
#     FLAG       returns
#     "dim"      N
#     "real"     1 if A is real, 0 otherwise
#     "notransp" A*X
#     "transp"   A'*X
#
#   C = NORMEST1(A,T) changes the number of columns in the iteration matrix
#   from the default 2.  Choosing T <= N/4 is recommended, otherwise it should
#   be cheaper to form the norm exactly from the elements of A, as is done
#   when N <= 4 or T == N.  If T < 0 then ABS(T) columns are used and trace
#   information is printed.  If T is given as the empty matrix [] then the
#   default T is used.
#
#   C = NORMEST1(A,T,X0) specifies a starting matrix X0 with columns of unit
#   1-norm and by default is random for T > 1.  If X0 is given as the empty
#   matrix [] then the default X0 is used.
#
#   C = NORMEST1(AFUN,T,X0,P1,P2,...) passes extra inputs P1,P2,... to
#   FEVAL(@AFUN,FLAG,X,P1,P2,...).
#
#   [C,V] = NORMEST1(A,...) and [C,V,W] = NORMEST1(A,...) also return vectors
#   V and W such that W = A*V and NORM(W,1) = C*NORM(V,1).
#
#   [C,V,W,IT] = normest1(A,...) also returns a vector IT such that
#   IT[1] is the number of iterations
#   IT[2] is the number of products of N-by-N by N-by-T matrices.
#   On average, IT[2] = 4.
#
#   Note: NORMEST1 uses random numbers generated by RAND.  If repeatable
#   results are required,  use RNG to control MATLAB's random number
#   generator state.
#
#   See also CONDEST, COND, NORM, RAND.

#   Subfunctions: MYSIGN, UNDUPLI, NORMAPP.

#   Reference:
#   [1] Nicholas J. Higham and Fran\c{c}oise Tisseur,
#       A Block Algorithm for Matrix 1-Norm Estimation
#       with an Application to 1-Norm Pseudospectra,
#       SIAM J. Matrix Anal. App. 21, 1185-1201, 2000.

#   Nicholas J. Higham
#   Copyright 1984-2012 The MathWorks, Inc.
function normest1(
      A,
      t::Int = 2,
      X = initialize_X(A,t)
   )

   @assert size(A,1) == size(A,2) "AbstractMatrix A must be square"

   A_is_real = isreal(A)
   Te = eltype(A)
   n = size(A,2)
   
   prnt = (t < 0)
   t = abs(t)
   
   #error(message("MATLAB:normest1:TOutOfRange"))
   (t < 1 || t > max(n,2)) && error("t must be a non-zero integer with magnitude <= size(A,2)")

   rpt_S = 0
   rpt_e = 0

   if t == n || n <= 4
      # Get full matrix
      Y = zeros(Te, n, n)
      X = Array{Te}(I, n, n)
      mul!(Y, A, X) # A_mul_B!(Y, A, X)

      # equivalent to `temp, m = sort( sum(abs(Y)) )` in MATLAB
      temp = sum(abs, Y, dims=1)
      m = sortperm(temp[:])
      temp = temp[:,m]

      est = temp[n] # `full(temp[n])` is unnecessary; elements of sparse matrices are not sparse in Julia
      v = zeros(Te, n)
      v[m[n]] = one(Te)
      w = Y[:,m[n]]

      iter = [0 1]
      #fprintf(getString(message("MATLAB:normest1:NoIterationNormComputedExactly")))
      prnt && println("No iterations: norm computed exactly")

      return est, v, w, iter
   end

   #error(message("MATLAB:normest1:WrongColNum", int2str( t )))
   (size(X,2) != t) && error("Number of columns of X must match t = $t, but size(X,2) = $(size(X,2)).")

   itmax = 5  # Maximum number of iterations.
   it = 0
   nmv = 0

   ind = zeros(Int, t)
   est_old = 0
   ind_hist = ind #TODO make sure this is a good initialization
   est_j = 0 #TODO make sure this is a good initialization

   # S_type = A_is_real ? Int : Te
   S = zeros(Te, n, t)
   SS = zeros(Te, t, t)

   Y = similar(S, Te)
   Z = similar(S, Te)
   S_old = similar(S, Te)

   while true
      it += 1

      # prnt && @show typeof(Y), typeof(A), typeof(X)
      mul!(Y, A, X) # A_mul_B!(Y, A, X)
      nmv += 1

      vals = sum(abs, Y, dims=1)
      m = sortperm(vals[:])
      m = m[t:-1:1]
      vals = vals[:,m]
      vals_ind = ind[m]
      est = vals[1]

      if est > est_old || it == 2
         est_j = vals_ind[1]
         w = Y[:,m[1]]
      end

      if prnt
      @printf("%g: ", it)
         for i = 1:t
            @printf(" (%g, %6.2e)", vals_ind[i], vals[i])
         end
      @printf("\n")
      end

      if it >= 2 && est <= est_old
         est = est_old
         info = 2
         break
      end
      est_old = est

      if it > itmax
         it = itmax
         info = 1
         break
      end

      S_old .= S
      S .= mysign.(Y)
      if A_is_real
         # SS = S_old'*S
         mul!(SS, adjoint(S_old), S) # Ac_mul_B!(SS, S_old, S)

         np = sum(x->x==Te(n), maximum(abs, SS, dims=1))
         if np == t
            info = 3
            break
         end

         # Now check/fix cols of S parallel to cols of S or S_old.
         S, r = undupli(S, S_old, prnt)
         rpt_S = rpt_S + r
      end

      # prnt && @show typeof(Z), typeof(A), typeof(X)
      mul!(Z, adjoint(A), S) # Ac_mul_B!(Z, A, S)
      nmv = nmv + 1

      # Faster version of `for i=1:n, Zvals[i] = norm(Z[i,:], inf); end`:
      Zvals = maximum(abs, Z, dims=2)

      if it >= 2
         if maximum(Zvals) == Zvals[est_j]
            info = 4
            break
         end
      end

      # m = sortperm(Zvals[:])
      # m = m[n:-1:1]
      m = sortperm(Zvals[:]; rev=true)
      imax = t; # Number of new unit vectors; may be reduced below (if it > 1).
      if it == 1
         ind = m[1:t]
         ind_hist = ind
      else
      #`in.(A,[B])` is equivalent to MATLAB's `ismember(A,B)`
      rep = sum(in.(m[1:t], [ind_hist])) # ismember(m[1:t], [ind_hist])
      rpt_e = rpt_e + rep
      if rep > 0 && prnt
          @printf("     rep e_j = %g\n",rep)
      end
      if rep == t
          info = 5
          break
      end
      j = 1
      for i = 1:t
         if j > n
            imax = i-1
            break
         end
         while any( ind_hist .== m[j] )
            j = j+1
            if j > n
               imax = i-1
               break
            end
         end
         if j > n
            break
         end
            ind[i] = m[j]
            j = j+1
         end
         ind_hist = [ind_hist; ind[1:imax]]
      end

      fill!(X, zero(Te)) #X = zeros(Te, n, t)
      for j=1:imax
         X[ind[j],j] = one(Te)
      end
   end

   if prnt
      if info == 1
         #fprintf(getString(message("MATLAB:normest1:TerminateIterationLimitReachedn")))
         println("normest1: terminate; iteration limit reached")
      elseif info == 2
         #fprintf(getString(message("MATLAB:normest1:TerminateEstimateNotIncreased")))
         println("normest1: terminate; estimate not increased")
      elseif info == 3
         #fprintf(getString(message("MATLAB:normest1:TerminateRepeatedSignMatrix")))
         println("normest1: terminate; repeated sign matrix")
      elseif info == 4
         #fprintf(getString(message("MATLAB:normest1:TerminatePowerMethodConvergenceTest")))
         println("normest1: terminate; power method convergence test")
      elseif info == 5
         #fprintf(getString(message("MATLAB:normest1:TerminateRepeatedUnitVectors")))
         println("normest1: terminate; repeated unit vectors")
      end
   end

   iter = [it; nmv]
   red = [rpt_S; rpt_e]

   v = zeros(Te, n)
   v[est_j] = one(Te)
   # if !prnt
   return est, v, w, iter
   # end

   # if A_is_real
   #    fprintf(getString(message("MATLAB:normest1:ParallelCols", sprintf("#g",red[1]))))
   # end
   # fprintf(getString(message("MATLAB:normest1:RepeatedUnitVectors", sprintf("#g",red[2]))))

end

#############################################################################
# Subfunctions.

#MYSIGN True sign function with MYSIGN(0) = 1.
@inline @fastmath mysign(x) = x == zero(x) ? one(x) : sign(x)

#INITIALIZE_x
function initialize_X(A, t::Int)
   Te = eltype(A)
   prnt = (t<0)
   t = abs(t)
   n = size(A,2)

   X = ones(Te, n, t)
   @views X[:,2:t] = rand(Te.(-1:2:1), n, t-1)
   X, r = undupli(X, Matrix{Te}(undef,0,0), false)
   X ./= n

   return X::Matrix{Te}
end

#UNDUPLI   Look for and replace columns of S parallel to other columns of S or
#          to columns of Sold.
function undupli(
      S::AbstractMatrix{Te},
      S_old::AbstractMatrix{Te},
      prnt::Bool
   ) where {Te}

   n, t = size(S)
   r = 0
   (t == 1) && return S, r

   if isempty(S_old) # Looking just at S.
      # W = [S[:,1] zeros(n,t-1)]
      W = zeros(Te,n,t)
      @views W[:,1] = S[:,1]
      jstart = 2
      last_col = 1
   else              # Looking at S and S_old.
      # W = [S_old zeros(n,t-1)]
      ncols = size(S_old,2)
      W = zeros(Te,n,t-1+ncols)
      @views W[:,1:ncols] = S_old
      jstart = 1
      last_col = t
   end

   for j=jstart:t
      rpt = 0
      while maximum(abs, @views S[:,j]'*W[:,1:last_col]) == Te(n)
         rpt = rpt + 1
         @views S[:,j] = rand(Te.(-1:2:1), n)
         if rpt > n/t
            break
         end
      end
      #fprintf(getString(message("MATLAB:normest1:UnduplicateRpt", sprintf("#g",rpt))))
      (prnt && rpt > 0) && print("normest1: unduplicate rpt: $rpt")

      r = r + sign(rpt)
      if j < t
         last_col = last_col + 1
         @views W[:,last_col] = S[:,j]
      end
   end

   return S, r
end