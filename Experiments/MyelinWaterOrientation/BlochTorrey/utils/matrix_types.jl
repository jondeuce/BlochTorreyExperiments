using LinearAlgebra, SparseArrays

####
#### InterleavedMatrix
####

struct InterleavedMatrix{T,AType,skip,isviews}
    A::AType
    InterleavedMatrix(A::AType, skip::Int) where {AType} = new{eltype(A), AType, skip, true}(A)
end
getskip(A::InterleavedMatrix{T,AType,skip}) where {T,AType,skip} = skip
Base.show(io::IO, A::InterleavedMatrix) = show(io, A.A)

Base.size(A::InterleavedMatrix, args...) = getskip(A) .* size(A.A, args...)
Base.:*(A::InterleavedMatrix, x) = mul!(similar(x),A,x)
LinearAlgebra.adjoint(A::InterleavedMatrix{T,AType,skip}) where {T,AType,skip} = InterleavedMatrix(A.A', skip)
LinearAlgebra.transpose(A::InterleavedMatrix{T,AType,skip}) where {T,AType,skip} = InterleavedMatrix(transpose(A.A), skip)

# In-place functions of the form f!(y, A, x) where y is modified in place
for f in [:(LinearAlgebra.mul!), :(LinearAlgebra.ldiv!)]
    @eval function $f(y,A::InterleavedMatrix{T,AType,skip,true},x) where {T,AType,skip}
        for start in 0:skip-1
            yv = view(y, firstindex(y,1) + start * stride(y,1) : skip * stride(y,1) : lastindex(y,1), :)
            xv = view(x, firstindex(x,1) + start * stride(x,1) : skip * stride(x,1) : lastindex(x,1), :)
            $f(yv, A.A, xv)
        end
        return y
    end
end

# Out-of-place functions of the form f(A, x) returning y of the same size and type as x
for f in [:(Base.:\)]
    @eval function $f(A::InterleavedMatrix{T,AType,skip,true},x) where {T,AType,skip}
        y = similar(x)
        for start in 0:skip-1
            yv = view(y, firstindex(y,1) + start * stride(y,1) : skip * stride(y,1) : lastindex(y,1), :)
            xv = view(x, firstindex(x,1) + start * stride(x,1) : skip * stride(x,1) : lastindex(x,1), :) |> collect # TODO: cholesky(A)\x fails without this
            copyto!(yv, $f(A.A, xv))
        end
        return y
    end
end

# Out-of-place functions of the form f(x, A) returning y of the same size and type as x
for f in [:(Base.:/)]
    @eval function $f(x,A::InterleavedMatrix{T,AType,skip,true}) where {T,AType,skip}
        y = similar(x)
        for start in 0:skip-1
            yv = view(y, firstindex(y,1) + start * stride(y,1) : skip * stride(y,1) : lastindex(y,1), :)
            xv = view(x, firstindex(x,1) + start * stride(x,1) : skip * stride(x,1) : lastindex(x,1), :) |> collect # TODO: cholesky(A)\x fails without this
            copyto!(yv, $f(xv, A.A))
        end
        return y
    end
end

####
#### BlockDiagonalMatrix
####

struct BlockDiagonalMatrix{T,AType}
    As::Vector{AType}
    sizes::Vector{NTuple{2,Int}}
    BlockDiagonalMatrix(As::AType...) where {AType} = new{eltype(AType), AType}([As...], [size.(As)...])
    BlockDiagonalMatrix(As::Vector{AType}) where {AType} = new{eltype(AType), AType}(As, size.(As))
end
Base.show(io::IO, A::BlockDiagonalMatrix) = (for A in A.As; show(io, A); end; nothing) 
Base.size(A::BlockDiagonalMatrix, args...) = reduce((x,y)->x.+y, size(A, args...) for A in A.As)
Base.:*(A::BlockDiagonalMatrix, x) = LinearAlgebra.mul!(similar(x),A,x)

Base.Matrix(A::BlockDiagonalMatrix{T,AType}) where {T,AType <: AbstractSparseArray} = blockdiag(A.As...)
function Base.Matrix(A::BlockDiagonalMatrix{T,AType}) where {T,AType}
    Af = zeros(T, size(A))
    start = 0
    for i in 1:length(A.As)
        (i > 1) && (start += A.sizes[i-1][2])
        ix = start .+ (1:A.sizes[i][2])
        Af[ix,ix] .= A.As[i]
    end
    return Af
end

# In-place functions of the form f!(y, A, x) where y is modified in place
for f in [:(LinearAlgebra.mul!), :(LinearAlgebra.ldiv!)]
    @eval function $f(y,A::BlockDiagonalMatrix,x)
        start = 0
        for i in 1:length(A.As)
            (i > 1) && (start += A.sizes[i-1][2])
            yv = view(y, firstindex(y,1) .+ stride(y,1) .* (start : start + A.sizes[i][2] - 1), :)
            xv = view(x, firstindex(x,1) .+ stride(x,1) .* (start : start + A.sizes[i][2] - 1), :)
            $f(yv, A.As[i], xv)
        end
        return y
    end
end

# Out-of-place functions of the form f(A, x) returning y of the same size and type as x
for f in [:(Base.:\)]
    @eval function $f(A::BlockDiagonalMatrix,x)
        y = similar(x)
        start = 0
        for i in 1:length(A.As)
            (i > 1) && (start += A.sizes[i-1][2])
            yv = view(y, firstindex(y,1) .+ stride(y,1) .* (start : start + A.sizes[i][2] - 1), :)
            xv = view(x, firstindex(x,1) .+ stride(x,1) .* (start : start + A.sizes[i][2] - 1), :) |> collect # TODO: cholesky(A)\x fails without this
            copyto!(yv, $f(A.As[i], xv))
        end
        return y
    end
end

# Out-of-place functions of the form f(x, A) returning y of the same size and type as x
for f in [:(Base.:/)]
    @eval function $f(x,A::BlockDiagonalMatrix)
        y = similar(x)
        start = 0
        for i in 1:length(A.As)
            (i > 1) && (start += A.sizes[i-1][2])
            yv = view(y, firstindex(y,1) .+ stride(y,1) .* (start : start + A.sizes[i][2] - 1), :)
            xv = view(x, firstindex(x,1) .+ stride(x,1) .* (start : start + A.sizes[i][2] - 1), :) |> collect # TODO: cholesky(A)\x fails without this
            copyto!(yv, $f(A.As[i], xv))
        end
        return y
    end
end

for f in [:transpose, :adjoint, :qr, :lu, :cholesky, :svd, :eigen, :hessenberg, :schur]
    # Unary LinearAlgebra module functions of the form f(A) returning a matrix A
    @eval LinearAlgebra.$f(A::BlockDiagonalMatrix) = BlockDiagonalMatrix(map($f, A.As))
end
