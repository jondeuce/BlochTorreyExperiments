function [ J ] = sparse( A )
%SPARSE Creat sparse matrix from BlochTorreyOp object A.

N = length(A);
[Nx,Ny,Nz] = deal(A.gsize(1),A.gsize(2),A.gsize(3));
[hx,hy,hz] = deal(A.h(1),A.h(2),A.h(3));
D = A.D;

% Initialize sparse matrix with diagonal term
if isscalar(A.Diag)
    J = A.Diag * speye(N,N);
else
    J = spdiags(A.Diag(:),0,N,N);
end

% Construct periodic Laplacian (without diagonal, which is included above)
if D ~= 0
    % 2nd difference block matrices, with diagonal zeroed
    Jx = (1/hx^2) * spdiags( repmat([1,0,1],Nx,1), [-1 0 1], Nx, Nx );
    Jy = (1/hy^2) * spdiags( repmat([1,0,1],Ny,1), [-1 0 1], Ny, Ny );
    Jz = (1/hz^2) * spdiags( repmat([1,0,1],Nz,1), [-1 0 1], Nz, Nz );

    % Set periodic BC's
    [ Jx(1,end), Jy(1,end), Jz(1,end), Jx(end,1), Jy(end,1), Jz(end,1) ] = ...
        deal(1/hx^2, 1/hy^2, 1/hz^2, 1/hx^2, 1/hy^2, 1/hz^2);

    [Ix,Iy,Iz] = deal( speye(Nx), speye(Ny), speye(Nz) );
    L = kron(Iz, kron(Iy, Jx)) + ...  % Laplacian matrix: d^2/dx^2 term
        kron(Iz, kron(Jy, Ix)) + ...  % d^2/dy^2 term
        kron(kron(Jz, Iy), Ix);       % d^2/dz^2 term
    
    % Add diffusive off-diagonal terms
    J = J + D * L;
end

end

