function [ J ] = sparse( A )
%SPARSE Creat sparse matrix from BlochTorreyOp object A.

N = length(A);
[Nx,Ny,Nz] = deal(A.gsize(1),A.gsize(2),A.gsize(3));

D = A.D;
[hx,hy,hz] = deal(A.h(1),A.h(2),A.h(3));
[ihx2,ihy2,ihz2] = deal( 1/hx^2, 1/hy^2, 1/hz^2 );

% 2nd difference block matrices, with diagonal zeroed
Jx = ihx2 * spdiags( repmat([1,0,1],Nx,1), [-1 0 1], Nx, Nx );
Jy = ihy2 * spdiags( repmat([1,0,1],Ny,1), [-1 0 1], Ny, Ny );
Jz = ihz2 * spdiags( repmat([1,0,1],Nz,1), [-1 0 1], Nz, Nz );

% Set periodic BC's
[ Jx(1,end), Jy(1,end), Jz(1,end), Jx(end,1), Jy(end,1), Jz(end,1) ] = ...
    deal(ihx2, ihy2, ihz2, ihx2, ihy2, ihz2);

% Initialize sparse matrix with diagonal term
J = spdiags(A.Diag(:),0,N,N);

if D ~= 0
    % Construct periodic Laplacian, with diagonal zeroed
    [Ix,Iy,Iz] = deal( speye(Nx), speye(Ny), speye(Nz) );
    L = kron(Iz, kron(Iy, Jx)) + ...  % Laplacian matrix: d^2/dx^2 term
        kron(Iz, kron(Jy, Ix)) + ...  % d^2/dy^2 term
        kron(kron(Jz, Iy), Ix);       % d^2/dz^2 term
    
    % Add diffusive off-diagonal terms
    J = J + D * L;
end

end

