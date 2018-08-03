module Normest1Fast

export normest1

function normest1(A, t=2, X=[], varargin=())
#NORMEST1 Estimate of 1-norm of matrix by block 1-norm power method.
#   C = NORMEST1(A) returns an estimate C of norm(A,1), where A is N-by-N.
#   A can be an explicit matrix or a function AFUN such that
#   FEVAL(@AFUN,FLAG,X) for the following values of
#     FLAG       returns
#     "dim"      N
#     "real"     1 if A is real, 0 otherwise
#     "notransp" A*X
#     "transp"   A"*X
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

   # Determine whether A is a matrix or a function.
   if isa(A, AbstractMatrix{<:AbstractFloat}) #isfloat(A) && ismatrix(A)
      n = maximum(size(A))
      A_is_real = isreal(A)
   elseif isa(A,Function)
      n = normapp(A,"dim",[],varargin)
      A_is_real = normapp(A,"real",[],varargin)
   else
      #error(message("MATLAB:normest1:ANotMatrixOrFunction"))
   end

   if isempty(t)
      t = 2
   end
   prnt = (t < 0)
   t = abs(t)
   if t != round(t) || t < 1 || t > max(n,2)
      #error(message("MATLAB:normest1:TOutOfRange"))
   end
   rpt_S = 0; rpt_e = 0

   if t == n || n <= 4
      if isa(A, AbstractMatrix{<:AbstractFloat}) #isfloat(A)
         Y = A
      else
         X = eye(n)
         Y = normapp(A,"notransp",X,varargin)
      end

      # equivalent to `temp, m = sort( sum(abs(Y)) )` in MATLAB
      temp = sum(abs, Y, 1)
      m = sortperm(temp[:])
      temp = temp[:,m]

      est = temp[n] # `full(temp[n])` is unnecessary; elements of sparse matrices are not sparse in Julia
      v = zeros(n,1)
      v[m[n]] = 1
      w = Y[:,m[n]]

      iter = [0 1]
      if prnt
         #fprintf(getString(message("MATLAB:normest1:NoIterationNormComputedExactly")))
      end
      return est, v, w, iter
   end

   if isempty(X)
      X = ones(n,t)
      @views X[:,2:t] .= mysign.(2*rand(n,t-1) - ones(n,t-1))
      X, r = undupli(X, [], prnt)
      X ./= n
   end

   if size(X,2) != t
     #error(message("MATLAB:normest1:WrongColNum", int2str( t )))
   end

   itmax = 5  # Maximum number of iterations.
   it = 0
   nmv = 0

   ind = zeros(t,1)
   S = zeros(n,t)
   est_old = 0
   est_j = 0 #TODO make sure this is a good initialization
   Y = similar(X)
   Z = similar(X)

   while true
       it += 1

       if isa(A, AbstractMatrix{<:AbstractFloat}) # isfloat(A)
          A_mul_B!(Y, A, X)
       else
          Y = normapp(A,"notransp",X,varargin)
       end
       nmv += 1

       vals = sum(abs, Y, 1)
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
          fprintf("%g: ", it)
          for i = 1:t
             fprintf(" (%g, %6.2e)", vals_ind[i], vals[i])
          end
          fprintf("\n")
       end

       if it >= 2 && est <= est_old
          est = est_old
          info = 2; break
       end
       est_old = est

       if it > itmax
          it = itmax
          info = 1
          break
       end

       S_old = S
       S = mysign.(Y)
       if A_is_real
           SS = S_old'*S
           np = sum(x->x==n, maximum(abs, SS, 1))
           if np == t
             info = 3
             break
          end
          # Now check/fix cols of S parallel to cols of S or S_old.
          S, r = undupli(S, S_old, prnt)
          rpt_S = rpt_S + r
       end

       if isa(A, AbstractMatrix{<:AbstractFloat}) # isfloat(A)
          At_mul_B!(Z, A, S)
       else
          Z = normapp(A,"transp",S,varargin)
       end
       nmv = nmv + 1

       # Faster version of `for i=1:n, Zvals[i] = norm(Z[i,:], inf); end`:
       Zvals = maximum(abs, Z, 2)

       if it >= 2
          if maximum(Zvals) == Zvals[est_j]
             info = 4
             break
          end
       end

       m = sortperm(Zvals[:])
       m = m[n:-1:1]
       imax = t; # Number of new unit vectors; may be reduced below (if it > 1).
       if it == 1
          ind = m[1:t]
          ind_hist = ind
       else
          #`in.(A,[B])` is equivalent to MATLAB's `ismember(A,B)`
          rep = sum(in.(m[1:t], [ind_hist])) # ismember(m[1:t], [ind_hist])
          rpt_e = rpt_e + rep
          if rep > 0 && prnt
             fprintf("     rep e_j = %g\n",rep)
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
              while any( ind_hist == m[j] )
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

       X = zeros(n,t)
       for j=1:imax
          X[ind[j],j] = 1
       end
   end

   if prnt
      if info == 1
         #fprintf(getString(message("MATLAB:normest1:TerminateIterationLimitReachedn")))
      elseif info == 2
         #fprintf(getString(message("MATLAB:normest1:TerminateEstimateNotIncreased")))
      elseif info == 3
         #fprintf(getString(message("MATLAB:normest1:TerminateRepeatedSignMatrix")))
      elseif info == 4
         #fprintf(getString(message("MATLAB:normest1:TerminatePowerMethodConvergenceTest")))
      elseif info == 5
         #fprintf(getString(message("MATLAB:normest1:TerminateRepeatedUnitVectors")))
      end
   end

   iter = [it; nmv]; red = [rpt_S; rpt_e]

   v = zeros(n,1); v[est_j] = 1
   if !prnt
      return est, v, w, iter
   end

   if A_is_real
      #fprintf(getString(message("MATLAB:normest1:ParallelCols", sprintf("#g",red[1]))))
   end
   #fprintf(getString(message("MATLAB:normest1:RepeatedUnitVectors", sprintf("#g",red[2]))))

end

#############################################################################
# Subfunctions.

function undupli(S, S_old, prnt)
#UNDUPLI   Look for and replace columns of S parallel to other columns of S or
#          to columns of Sold.

   n, t = size(S)
   r = 0
   if t == 1
      return S, r
   end

   if isempty(S_old) # Looking just at S.
      W = [S[:,1] zeros(n,t-1)]
      jstart = 2
      last_col = 1
   else              # Looking at S and S_old.
      W = [S_old zeros(n,t-1)]
      jstart = 1
      last_col = t
   end

   for j=jstart:t
       rpt = 0
       while maximum(abs, @views S[:,j]'*W[:,1:last_col] ) == n
             rpt = rpt + 1
             @views S[:,j] .= mysign.(2*rand(n,1) - ones(n,1))
             if rpt > n/t
                break
             end
       end
       if prnt && rpt > 0
          #fprintf(getString(message("MATLAB:normest1:UnduplicateRpt", sprintf("#g",rpt))))
       end
       r = r + sign(rpt)
       if j < t
          last_col = last_col + 1
          @views W[:,last_col] = S[:,j]
       end
   end

   return S, r

end

function mysign(x::T) where {T}
   #MYSIGN      True sign function with MYSIGN(0) = 1.

   return ifelse(x < zero(T), -one(T), one(T))

end

function normapp(afun,flag,x,varargin=())
#NORMAPP   Call matrix operator and error gracefully.
#   NORMAPP(AFUN,FLAG,X) calls matrix operator AFUN with flag
#   FLAG and matrix X.
#   NORMAPP(AFUN,FLAG,X,...) allows extra arguments to
#   AFUN(FLAG,X,...).
#   NORMAPP is designed for use by NORMEST1.

   y = try
      afun(flag,x,varargin...)
   catch ME
      #error(message("MATLAB:normest1:Failure", string( afun ), ME.message))
      rethrow(ME)
   end

   if isequal(flag,"notransp") || isequal(flag,"transp")
      if !isequal(size(y),size(x))
         #error(message("MATLAB:normest1:MatrixSizeMismatchFlag", string( afun ), size( x, 1 ), size( x, 2 ), flag))
      end
   end

   if isequal(flag,"dim")
      if y != round(y) || y < 0
         #error(message("MATLAB:normest1:NegInt", string( afun ), flag))
      end
   end

   if isequal(flag,"real")
      if y != 0 && y != 1
         #error(message("MATLAB:normest1:Not0or1", string( afun ), flag))
      end
   end

   return y

end

# test `afun` where a matrix `A` is wrapped and given as a varargin parameter
function test_afun(flag,x,A)
   if flag == "notransp"
      return A*x
   elseif flag == "transp"
      return A'*x
   elseif flag == "dim"
      return size(A,1)
   elseif flag == "real"
      return isreal(A)
   else
      error("Unknown flag: $flag")
   end
end

# test `afun` where stiffness matrix `K` and factored mass matrix `Mfact` are
# wrapped and given as a varargin parameters
function mass_and_stifness_afun(flag,x,K,Mfact)
   if flag == "notransp" # inv(M)*K*x
      tmp = similar(x)
      A_mul_B!(tmp, K, x)
      return Mfact\tmp
   elseif flag == "transp" # (inv(M)*K)'*x == K'*inv(M)*x (M is symmetric)
      tmp = Mfact\x
      return At_mul_B(K, tmp)
   elseif flag == "dim"
      return size(K,1)
   elseif flag == "real"
      return isreal(K) && isreal(M)
   else
      error("Unknown flag: $flag")
   end
end

function profile_normest1(;Asize::Int = 5000, Nloops::Int = 100, prnt::Bool = false)
   A = randn(Asize,Asize)
   normest1(A) # dry run to precompile

   Profile.clear()
   Profile.init(;n=10_000_000)
   @profile for i = 1:Nloops
      normest1(A)[1]
   end
   prnt && Profile.print()

   return nothing
end

end # module Normest1Fast

nothing