# Wrap the action of Mfact\K in a LinearMap
function Minv_K_mul_u!(Y, X, K, Mfact)
   A_mul_B!(Y, K, X)
   copy!(Y, Mfact\Y)
   return Y
end
function Kt_Minv_mul_u!(Y, X, K, Mfact)
   At_mul_B!(Y, K, Mfact\X)
   return Y
end
function get_mass_and_stifness_map(K, Mfact)
   @assert size(K) == size(Mfact)
   fwd_mul! = (Y, X) -> Minv_K_mul_u!(Y, X, K, Mfact);
   trans_mul! = (Y, X) -> Kt_Minv_mul_u!(Y, X, K, Mfact);
   return LinearMap(fwd_mul!, trans_mul!, size(K)...;
      ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)
end

#TODO: Probably don't need to define these; would only be used for normest1
# which is definitely not a bottleneck, and this clearly could come back to bite
# me at some unknown time...
# import Base.LinAlg: A_mul_B!, At_mul_B!, Ac_mul_B!
# A_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.f!(Y,X);
# At_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.fc!(Y,X);
# Ac_mul_B!(Y::AbstractMatrix, A::FunctionMap, X::AbstractMatrix) = A.fc!(Y,X);

# Custom norm for calling expmv
expmv_norm(x::AbstractVector, p::Real=2, args...) = Base.norm(x, p, args...) #fallback
function expmv_norm(A, p::Real=1, t::Int=10)
    if p == 1
        return normest1(A, t)[1]
    elseif p == Inf
        return normest1(A', t)[1]
    else
        error("Only p=1 or p=Inf supported")
    end
end
