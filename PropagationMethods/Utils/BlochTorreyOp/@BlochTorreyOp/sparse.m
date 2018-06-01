function [ J ] = sparse( A )
%SPARSE Creat sparse matrix from BlochTorreyOp object A.

N = length(A);
[Nx,Ny,Nz] = deal(A.gsize(1),A.gsize(2),A.gsize(3));
[hx,hy,hz] = deal(A.h(1),A.h(2),A.h(3));

toDiag = @(x) spdiags(x,0,N,N);

% Initialize sparse matrix with diagonal Gamma term
if isscalar(A.Gamma)
    J = -A.Gamma * speye(N,N);
else
    J = toDiag(-A.Gamma(:));
end

if isequal(A.D, 0)
    return
end

% Construct periodic Laplacian via Kronecker product of 2nd difference block matrices
Jx = spdiags( (1/hx^2) * repmat([1,-2,1],Nx,1), [-1 0 1], Nx, Nx );
Jy = spdiags( (1/hy^2) * repmat([1,-2,1],Ny,1), [-1 0 1], Ny, Ny );
Jz = spdiags( (1/hz^2) * repmat([1,-2,1],Nz,1), [-1 0 1], Nz, Nz );

% Set periodic BC's
[ Jx(1,end), Jy(1,end), Jz(1,end), Jx(end,1), Jy(end,1), Jz(end,1) ] = ...
    deal(1/hx^2, 1/hy^2, 1/hz^2, 1/hx^2, 1/hy^2, 1/hz^2);

% Sparse identity matrices along each dimension
[Ix,Iy,Iz] = deal( speye(Nx), speye(Ny), speye(Nz) );

% Periodic Laplacian
L = kron(Iz, kron(Iy, Jx)) + ...  % Laplacian matrix: d^2/dx^2 term
    kron(Iz, kron(Jy, Ix)) + ...  % d^2/dy^2 term
    kron(kron(Jz, Iy), Ix);       % d^2/dz^2 term

if isscalar(A.D)
    % Add isotropic diffusive term
    J = J + A.D * L;
else
    % Diffusion operator is:
    %     Lap(D*grad(u)) = D*Lap(u) + dot(grad(D),grad(u))
    % The first ("isotropic diffusion") term is simple. The second
    % ("flux diffusion") term needs to be symmetrised, averaging
    % forward and backward differences on D and u
    
    % "Isotropic diffusion" term is simply multiplying pointwise by D, i.e.
    % multiplication with a diagonal matrix on the left
    Dv = A.D(:);
    D  = spdiags( Dv, 0, N, N );
    J  = J + D * L;
    
    % "Flux diffusion" is more complex; first, need forward/backward
    % difference operators
    Dxb = spdiags( (1/hx) * repmat([-1,1],Nx,1), [-1 0], Nx, Nx );
    Dyb = spdiags( (1/hy) * repmat([-1,1],Ny,1), [-1 0], Ny, Ny );
    Dzb = spdiags( (1/hz) * repmat([-1,1],Nz,1), [-1 0], Nz, Nz );
    Dxf = spdiags( (1/hx) * repmat([-1,1],Nx,1), [ 0 1], Nx, Nx );
    Dyf = spdiags( (1/hy) * repmat([-1,1],Ny,1), [ 0 1], Ny, Ny );
    Dzf = spdiags( (1/hz) * repmat([-1,1],Nz,1), [ 0 1], Nz, Nz );
    
    % Periodic boundary conditions
    [ Dxb(1,end), Dyb(1,end), Dzb(1,end), Dxf(end,1), Dyf(end,1), Dzf(end,1) ] = ...
        deal(-1/hx, -1/hy, -1/hz, 1/hx, 1/hy, 1/hz);
    
    % Construct full matrices via Kronecker product, and add averaged
    % forward/backward difference "flux diffusion" terms one by one (just
    % to save memory)
    Dxb = kron(Iz, kron(Iy, Dxb)); % Forward difference matrix: d/dx
    Dxf = kron(Iz, kron(Iy, Dxf)); % Forward difference matrix: d/dx
    J = J + (1/2) * ( toDiag(Dxf*Dv) * Dxb + toDiag(Dxb*Dv) * Dxf );
    clear Dxb Dxf
    
    Dyb = kron(Iz, kron(Dyb, Ix)); % backward d/dy
    Dyf = kron(Iz, kron(Dyf, Ix)); % forward d/dy
    J = J + (1/2) * ( toDiag(Dyf*Dv) * Dyb + toDiag(Dyb*Dv) * Dyf );
    clear Dyb Dyf
    
    Dzb = kron(kron(Dzb, Iy), Ix); % backward d/dz
    Dzf = kron(kron(Dzf, Iy), Ix); % forward d/dz
    J = J + (1/2) * ( toDiag(Dzf*Dv) * Dzb + toDiag(Dzb*Dv) * Dzf );
    clear Dzb Dzf
end

end

