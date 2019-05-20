using Test
using LinearAlgebra, LinearMaps, FFTW
using Random
using DrWatson: @ntuple, @dict

const Mat2D = AbstractMatrix
const Mat3D = AbstractArray{T,3} where T

randnc(n...) = complex.(randn(n...), randn(n...));
randc(n...) = complex.(rand(n...), rand(n...));
toherm(A) = (A + A')/2;
makeposdef(n) = toherm(randc(n, n)) + n*I
eyelike(B) = Matrix{eltype(B)}(I, size(B)...);

# Kronecker sum: A ⊕ B = A ⊗ I_B + I_A ⊗ B
const ⊗ = kron
ksum(A, B) = A ⊗ eyelike(B) + eyelike(A) ⊗ B;
const ⊕ = ksum

# Kronecker-vector product: (A ⊗ B) * vec(X) == vec(B * X * A^T)
kronv(A::Mat2D, B::Mat2D, X::Mat2D) = vec(B * transpose(A * transpose(X)))
# kronv(A::Mat2D, B::Mat2D, X::Mat2D) = vec(B * X * transpose(A)) # equivalent

# Triple Kronecker-vector product: (A ⊗ B ⊗ C) * vec(X)
function kronv(A::Mat2D, B::Mat2D, C::Mat2D, X::Mat3D)
    n = (A = size(A,2), B = size(B,2), C = size(C,2))
    
    # # NOTE: The specific permutations don't matter, only that the correct
    # #       dimension is first permuted to the front and then permuted back
    # Y = copy(X)
    # # for j in 1:size(X,2), i in 1:size(X,1)
    # #     @views mul!(Y[i,j,:], A, X[i,j,:])
    # # end
    # # for k in 1:size(X,3)
    # #     @views mul!(Y[:,:,k], A, X[:,:,k])
    # # end
    # for j in 1:size(X,2)
    #     @views mul!(Y[:,j,:]', A, X[:,j,:]')
    # end
    # # for i in 1:size(X,1)
    # #     @views mul!(Y[i,:,:]', A, X[i,:,:]')
    # # end

    # Y = permutedims(Y, (2,3,1)) # (2,3,1) then (2,3,1); equiv. to (3,1,2)
    # Y = reshape(Y, (n.B, length(Y) ÷ n.B))
    # Y = B * Y
    # Y = reshape(Y, (n.B, n.A, n.C))
    
    # Y = permutedims(Y, (3,1,2)) # (2,3,1) then (1,2,3); equiv. to (2,3,1)
    # Y = reshape(Y, (n.C, length(Y) ÷ n.C))
    # Y = C * Y
    # Y = reshape(Y, (n.C, n.B, n.A)) # (1,2,3) is a no-op

    # return Y
    
    # NOTE: The specific permutations don't matter, only that the correct
    #       dimension is first permuted to the front and then permuted back
    
    # First: sum(Alk * Xijk)
    X = permutedims(X, (3,1,2))
    # X = PermutedDimsArray(X, (3,1,2))
    X = reshape(X, (n.A, length(X) ÷ n.A))
    X = A * X
    X = reshape(X, (n.A, n.C, n.B))

    X = permutedims(X, (3,1,2)) # (2,3,1) then (2,3,1); equiv. to (3,1,2)
    # X = PermutedDimsArray(X, (3,1,2)) # (2,3,1) then (2,3,1); equiv. to (3,1,2)
    X = reshape(X, (n.B, length(X) ÷ n.B))
    X = B * X
    X = reshape(X, (n.B, n.A, n.C))
    
    X = permutedims(X, (3,1,2)) # (2,3,1) then (1,2,3); equiv. to (2,3,1)
    # X = PermutedDimsArray(X, (3,1,2)) # (2,3,1) then (1,2,3); equiv. to (2,3,1)
    X = reshape(X, (n.C, length(X) ÷ n.C))
    X = C * X
    X = reshape(X, (n.C, n.B, n.A)) # (1,2,3) is a no-op
    
    return X
end

####
#### Testing
####

function test_kron_props()
    A = randnc(5,5);
    B = randnc(4,4);
    C = randnc(3,3);

    # Commutivity of triply kron sum
    @test (A ⊕ B) ⊕ C ≈ A ⊕ (B ⊕ C)

    # Triple kron sum expansion
    @test A ⊕ (B ⊕ C) ≈ begin
        IA, IB, IC = eyelike(A), eyelike(B), eyelike(C)
        A ⊗ IB ⊗ IC + IA ⊗ B ⊗ IC + IA ⊗ IB ⊗ C
    end

    nothing
end

function test_kronv_2d(n = (A=10, B=8), t = 0.1)
    A, B = randnc(n.A, n.A), randnc(n.B, n.B) # makeposdef
    EA, EB = map(X -> exp(-t*X), (A, B))
    
    # Test identity: exp(A ⊕ B) = exp(A) ⊗ exp(B)
    C = A ⊕ B;
    EC = exp(-t*C);
    @test EA ⊗ EB ≈ EC

    # Test action of exp(A ⊕ B):
    X = randnc(n.B, n.A)
    @test vec(kronv(EA, EB, X)) ≈ EC * vec(X)

    out = @ntuple A B C EA EB EC
    return out
end

function test_kronv_3d(n = (A=6, B=5, C=4), t = 0.1)
    A, B, C = randnc(n.A, n.A), randnc(n.B, n.B), randnc(n.C, n.C) # makeposdef
    EA, EB, EC = map(X -> exp(-t*X), (A, B, C))
    
    # Test identity: exp(A ⊕ B ⊕ C) = exp(A) ⊗ exp(B) ⊗ exp(C)
    D = A ⊕ B ⊕ C
    ED = exp(-t*D)
    @test EA ⊗ EB ⊗ EC ≈ ED

    # Test action of exp(A ⊕ B ⊕ C):
    X = randnc(n.C, n.B, n.A)
    @test vec(kronv(EA, EB, EC, X)) ≈ ED * vec(X)

    return nothing
end

# @testset begin
#     test_kron_props();
#     test_kronv_2d();
#     test_kronv_3d();
# end

####
#### Benchmarking
####

function benchmark_kronv_2d(n = (A=15, B=15), D = 500.0, t = 1.0)
    Random.seed!(0)
    X = randn(n.A, n.B)
    Lx, Ly = map(Matrix, make_laps(n))

    println("Kronecker Exponential-vector product")
    @time begin
        @time Ex, Ey = map(L -> exp(t.*D.*L), (Lx, Ly))
        @time Y1 = vec(kronv(Ey, Ex, X))
    end

    println("Gaussian convolution approximation")
    @time begin
        @time begin
            xs = CartesianIndices((-n.A÷2 : n.A÷2, -n.B÷2 : n.B÷2))
            Gxy = [exp(-(x[1]^2 + x[2]^2) / (4*D*t)) for x in xs]
            Gxy .= (g -> g >= eps() ? g : zero(g)).(Gxy) # remove subnormals
            Gxy ./= sum(Gxy) # normalize
        end
        @time Y3 = vec(ifft(fft(Gxy) .* fft(X)))
    end
    @show minimum(Gxy)
    @show maximum(abs, Y1 - Y3)

    if length(X) <= 1000
        println("Full Exponential-vector product")
        Lyx = Ly ⊕ Lx
        @time begin
            @time Eyx = exp(t.*D.*Lyx)
            @time Y2 = Eyx * vec(X)
        end
        display(@test Y1 ≈ Y2)
    end

    return nothing
end


function benchmark_kronv_3d(n = (A=9, B=9, C=9), D = 500.0, t = 1.0)
    Random.seed!(0)
    X = randn(n.A, n.B, n.C)
    Lx, Ly, Lz = map(Matrix, make_laps(n))

    (n.A == n.B) && display(@test Lx == Ly)
    (n.B == n.C) && display(@test Ly == Lz)

    println("Kronecker Exponential-vector product")
    @time begin
        @time Ex, Ey, Ez = map(L -> exp(t.*D.*L), (Lx, Ly, Lz))
        @time Y1 = vec(kronv(Ez, Ey, Ex, X))
    end

    println("Gaussian convolution approximation")
    @time begin
        @time begin
            xs = CartesianIndices((-n.A÷2 : n.A÷2, -n.B÷2 : n.B÷2, -n.C÷2 : n.C÷2))
            Gxyz = [exp(-(x[1]^2 + x[2]^2 + x[3]^2) / (4*D*t)) for x in xs]
            Gxyz .= (g -> g >= eps() ? g : zero(g)).(Gxyz) # remove subnormals
            Gxyz ./= sum(Gxyz) # normalize
        end
        @time Y3 = vec(ifft(fft(Gxyz) .* fft(X)))
    end
    @show minimum(Gxyz)
    @show maximum(abs, Y1 - Y3)

    if length(X) <= 1000
        println("Full Exponential-vector product")
        Lzyx = Lz ⊕ Ly ⊕ Lx
        @time begin
            @time Ezyx = exp(t.*D.*Lzyx)
            @time Y2 = Ezyx * vec(X)
        end
        display(@test Y1 ≈ Y2)
    end

    return nothing
end

####
#### Utility functions
####

function make_laps(n::NamedTuple{(:A,:B)} = (A=10, B=10))
    kwargs = (issymmetric = true, ishermitian = true, isposdef = false)
    return (Lx = LinearMap((y,x) -> lap_x!(reshape(y, (n.A, 1)), reshape(x, (n.A, 1))), n.A; kwargs...),
            Ly = LinearMap((y,x) -> lap_y!(reshape(y, (1, n.B)), reshape(x, (1, n.B))), n.B; kwargs...))
end

function make_laps(n::NamedTuple{(:A,:B,:C)} = (A=10, B=10, C=10))
    kwargs = (issymmetric = true, ishermitian = true, isposdef = false)
    return (Lx = LinearMap((y,x) -> lap_x!(reshape(y, (n.A, 1, 1)), reshape(x, (n.A, 1, 1))), n.A; kwargs...),
            Ly = LinearMap((y,x) -> lap_y!(reshape(y, (1, n.B, 1)), reshape(x, (1, n.B, 1))), n.B; kwargs...),
            Lz = LinearMap((y,x) -> lap_z!(reshape(y, (1, 1, n.C)), reshape(x, (1, 1, n.C))), n.C; kwargs...))
end

function lap_x!(Y::Mat2D, X::Mat2D)
    @assert size(Y) == size(X)
    Nx, Ny = size(X)
    @inbounds for j in 1:Ny, i in 1:Nx
        Y[i,j] = X[mod1(i-1, Nx), j] - 2*X[i,j] + X[mod1(i+1, Nx), j]
    end
    return Y
end
lap_x(X::Mat2D) = lap_x!(similar(X), X)

function lap_y!(Y::Mat2D, X::Mat2D)
    @assert size(Y) == size(X)
    Nx, Ny = size(X)
    @inbounds for j in 1:Ny, i in 1:Nx
        Y[i,j] = X[i, mod1(j-1, Ny)] - 2*X[i,j] + X[i, mod1(j+1, Ny)]
    end
    return Y
end
lap_y(X::Mat2D) = lap_y!(similar(X), X)

function lap_x!(Y::Mat3D, X::Mat3D)
    @assert size(Y) == size(X)
    Nx, Ny, Nz = size(X)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Y[i,j,k] = X[mod1(i-1, Nx), j, k] - 2*X[i,j,k] + X[mod1(i+1, Nx), j, k]
    end
    return Y
end
lap_x(X::Mat3D) = lap_x!(similar(X), X)

function lap_y!(Y::Mat3D, X::Mat3D)
    @assert size(Y) == size(X)
    Nx, Ny, Nz = size(X)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Y[i,j,k] = X[i, mod1(j-1, Ny), k] - 2*X[i,j,k] + X[i, mod1(j+1, Ny), k]
    end
    return Y
end
lap_y(X::Mat3D) = lap_y!(similar(X), X)

function lap_z!(Y::Mat3D, X::Mat3D)
    @assert size(Y) == size(X)
    Nx, Ny, Nz = size(X)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Y[i,j,k] = X[i, j, mod1(k-1, Nz)] - 2*X[i,j,k] + X[i, j, mod1(k+1, Nz)]
    end
    return Y
end
lap_z(X::Mat3D) = lap_z!(similar(X), X)
