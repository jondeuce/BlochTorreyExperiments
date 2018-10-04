# ============================================================================ #
# Speed testing for Cholesky vs. LU decomposition. Naively, Cholesky should be
# much faster and so there shouldn't be much need for testing. However, there is
# no inplace A_ldiv_B! for Cholesky factors, so far as I can tell. So, this is
# a quick test to see. It seems that Cholesky decomposition is still very much
# faster, as the copy is irrelevant when compared to the time saved on the
# matrix multiplies.
# ============================================================================ #

# ---------------------------------------------------------------------------- #
# Cholesky decomposition
# ---------------------------------------------------------------------------- #
function M_ldiv_K_mul!(du::AbstractVector{Tv},
                       M::SparseArrays.CHOLMOD.Factor{Tv},
                       K::SparseMatrixCSC{Tv,Ti},
                       u::Vector{Tv}) where {Tv,Ti}
    A_mul_B!(du, K, u)
    copy!(du, M\du)
end

# ---------------------------------------------------------------------------- #
# LU decomposition, with and without using a buffer
# ---------------------------------------------------------------------------- #
function M_ldiv_K_mul!(du::AbstractVector{Tv},
                       M::SparseArrays.UMFPACK.UmfpackLU{Tv,Ti},
                       K::SparseMatrixCSC{Tv,Ti},
                       u::Vector{Tv}) where {Tv,Ti}
    A_mul_B!(du, K, u)
    A_ldiv_B!(du, M, copy(du))
end

function M_ldiv_K_mul!(du::AbstractVector{Tv},
                       M::SparseArrays.UMFPACK.UmfpackLU{Tv,Ti},
                       K::SparseMatrixCSC{Tv,Ti},
                       u::Vector{Tv},
                       buffer::Vector{Tv}) where {Tv,Ti}
    A_mul_B!(buffer, K, u)
    A_ldiv_B!(du, M, buffer)
end
