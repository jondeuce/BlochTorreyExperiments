function [ J ] = sparse( A )
%SPARSE Creat sparse matrix from BlochTorreyOp object A.

N = length(A);
[Nx,Ny,Nz] = deal(A.gsize(1), A.gsize(2), A.gsize(3));
[hx,hy,hz] = deal(A.h(1), A.h(2), A.h(3));
[Ix,Iy,Iz] = deal(speye(Nx), speye(Ny), speye(Nz)); % Sparse identity matrices along each dimension

toDiag = @(x) spdiags(x,0,N,N);

% Initialize sparse matrix with diagonal Gamma term
if isscalar(A.Gamma)
    J = -A.Gamma * speye(N,N);
else
    J = toDiag(-vec(A.Gamma));
end

% If the diffusivity is zero, we are done
if isequal(A.D, 0)
    return
end

if isscalar(A.D)
    % Construct periodic Laplacian via Kronecker product of 2nd difference block matrices
    Jx = spdiags( repmat( (1/hx^2) * [1,-2,1], Nx, 1), [-1 0 1], Nx, Nx );
    Jy = spdiags( repmat( (1/hy^2) * [1,-2,1], Ny, 1), [-1 0 1], Ny, Ny );
    Jz = spdiags( repmat( (1/hz^2) * [1,-2,1], Nz, 1), [-1 0 1], Nz, Nz );
    
    % Set periodic BC's
    [ Jx(1,end), Jy(1,end), Jz(1,end), Jx(end,1), Jy(end,1), Jz(end,1) ] = ...
        deal(1/hx^2, 1/hy^2, 1/hz^2, 1/hx^2, 1/hy^2, 1/hz^2);
    
    % Periodic Laplacian
    L = kron(Iz, kron(Iy, Jx)) + ...  % Laplacian matrix: d^2/dx^2 term
        kron(Iz, kron(Jy, Ix)) + ...  % d^2/dy^2 term
        kron(kron(Jz, Iy), Ix);       % d^2/dz^2 term

    % Add isotropic diffusive term
    J = J + A.D * L;
else
    % Diffusion operator is:
    %       div(D*grad(u)) = sum_j d/dj(D*du/dj)
    % Let Fj be the forward finite difference matrix. Then, -Fj^T is the
    % backward finite difference matrix. Using backward differences for the
    % divergence and forward differences for the gradient, we have:
    %       div(D*grad(u)) = sum_j -Fj^T*diag(D)*Fj*u
    
    % Construct forward difference matrices (backwards: negative transpose)
    % and forward averaging matrices (backwards: transpose)
    Ax = spdiags( repmat( (0.5/hx) .* [ 1,1], Nx, 1), [0,1], Nx, Nx );
    Ay = spdiags( repmat( (0.5/hy) .* [ 1,1], Ny, 1), [0,1], Ny, Ny );
    Az = spdiags( repmat( (0.5/hz) .* [ 1,1], Nz, 1), [0,1], Nz, Nz );
    Fx = spdiags( repmat( (1.0/hx) .* [-1,1], Nx, 1), [0,1], Nx, Nx );
    Fy = spdiags( repmat( (1.0/hy) .* [-1,1], Ny, 1), [0,1], Ny, Ny );
    Fz = spdiags( repmat( (1.0/hz) .* [-1,1], Nz, 1), [0,1], Nz, Nz );
    
    % Periodic boundary conditions
    [ Fx(end,1), Fy(end,1), Fz(end,1) ] = deal(1.0/hx, 1.0/hy, 1.0/hz);
    [ Ax(end,1), Ay(end,1), Az(end,1) ] = deal(0.5/hx, 0.5/hy, 0.5/hz);
    
    % Construct full matrices via Kronecker product, and add averaged
    % forward/backward difference "flux diffusion" terms one by one (just
    % to save memory)
    if isempty(A.mask)
        Fx = kron(Iz, kron(Iy, Fx)); % Forward difference matrix: d/dx
        Ax = kron(Iz, kron(Iy, Ax)); % Forward averaging matrix: x-direction
        J = J + ( toDiag(Ax * vec(A.D)) * Fx + toDiag(Ax' * vec(A.D)) * Fx' );
        clear Fx Ax
        
        Fy = kron(Iz, kron(Fy, Ix)); % forward d/dy
        Ay = kron(Iz, kron(Ay, Ix)); % forward averaging in y
        J = J + ( toDiag(Ay * vec(A.D)) * Fy + toDiag(Ay' * vec(A.D)) * Fy' );
        clear Fy Ay
        
        Fz = kron(kron(Fz, Iy), Ix); % forward d/dz
        Az = kron(kron(Az, Iy), Ix); % forward averaging in z
        J = J + ( toDiag(Az * vec(A.D)) * Fz + toDiag(Az' * vec(A.D)) * Fz' );
        clear Fz Az
    else
        Fx = kron(Iz, kron(Iy, Fx)); % Forward difference matrix: d/dx
        Ax = kron(Iz, kron(Iy, Ax)); % Forward averaging matrix: x-direction
        Mx = (A.mask ~= circshift(A.mask, -1, 1)); % forward mask
        J = J + toDiag(maskRows(Ax, Mx) * vec(A.D)) * maskRows(Fx, Mx);
        Mx = (A.mask ~= circshift(A.mask, +1, 1)); % backward mask
        J = J + toDiag(maskRows(Ax', Mx) * vec(A.D)) * maskRows(Fx', Mx);
        clear Fx Ax Mx

        Fy = kron(Iz, kron(Fy, Ix)); % forward d/dy
        Ay = kron(Iz, kron(Ay, Ix)); % forward averaging in y
        My = (A.mask ~= circshift(A.mask, -1, 2)); % forward mask
        J = J + toDiag(maskRows(Ay, My) * vec(A.D)) * maskRows(Fy, My);
        My = (A.mask ~= circshift(A.mask, +1, 2)); % backward mask
        J = J + toDiag(maskRows(Ay', My) * vec(A.D)) * maskRows(Fy', My);
        clear Fy Ay My
        
        Fz = kron(kron(Fz, Iy), Ix); % forward d/dz
        Az = kron(kron(Az, Iy), Ix); % forward averaging in z
        Mz = (A.mask ~= circshift(A.mask, -1, 3)); % forward mask
        J = J + toDiag(maskRows(Az, Mz) * vec(A.D)) * maskRows(Fz, Mz);
        Mz = (A.mask ~= circshift(A.mask, +1, 3)); % backward mask
        J = J + toDiag(maskRows(Az', Mz) * vec(A.D)) * maskRows(Fz', Mz);
        clear Fz Az Mz
    end
end

end

function mask = maskRows(mask, b)
    if isempty(mask) || isempty(b); return; end
    mask(b, :) = 0;
end

% -------------------------------------------------- %
% ---- Forward grad/backward divergence version ---- %
% -------------------------------------------------- %

% % Diffusion operator is:
% %       div(D*grad(u)) = sum_j d/dj(D*du/dj)
% % Let Fj be the forward finite difference matrix. Then, -Fj^T is the
% % backward finite difference matrix. Using backward differences for the
% % divergence and forward differences for the gradient, we have:
% %       div(D*grad(u)) = sum_j -Fj^T*diag(D)*Fj*u
% 
% % "Isotropic diffusion" term is simply multiplying pointwise by D, i.e.
% % multiplication with a diagonal matrix on the left
% D  = toDiag(vec(A.D));
% 
% % Construct forward difference matrices (backwards: negative transpose)
% Bx = spdiags( (1/hx) * repmat([-1,1],Nx,1), [-1 0], Nx, Nx );
% By = spdiags( (1/hy) * repmat([-1,1],Ny,1), [-1 0], Ny, Ny );
% Bz = spdiags( (1/hz) * repmat([-1,1],Nz,1), [-1 0], Nz, Nz );
% Fx = spdiags( (1/hx) * repmat([-1,1],Nx,1), [ 0 1], Nx, Nx );
% Fy = spdiags( (1/hy) * repmat([-1,1],Ny,1), [ 0 1], Ny, Ny );
% Fz = spdiags( (1/hz) * repmat([-1,1],Nz,1), [ 0 1], Nz, Nz );
% 
% % Periodic boundary conditions
% [ Fx(end,1), Fy(end,1), Fz(end,1) ] = deal( 1/hx,  1/hy,  1/hz);
% [ Bx(1,end), By(1,end), Bz(1,end) ] = deal(-1/hx, -1/hy, -1/hz);
% 
% % Construct full matrices via Kronecker product, and add averaged
% % forward/backward difference "flux diffusion" terms one by one (just
% % to save memory)
% Fx = kron(Iz, kron(Iy, Fx)); % Forward difference matrix: d/dx
% J = J - ( Fx' * D * Fx );
% clear Fx
% 
% Fy = kron(Iz, kron(Fy, Ix)); % forward d/dy
% J = J - ( Fy' * D * Fy );
% clear Fy
% 
% Fz = kron(kron(Fz, Iy), Ix); % forward d/dz
% J = J - ( Fz' * D * Fz );
% clear Fz


% --------------------------------- %
% ---- Symmetrized Lap version ---- %
% --------------------------------- %

% % Diffusion operator is:
% %     Lap(D*grad(u)) = D*Lap(u) + dot(grad(D),grad(u))
% % The first ("isotropic diffusion") term is simple. The second
% % ("flux diffusion") term needs to be symmetrised, averaging
% % forward and backward differences on D and u
% 
% % "Isotropic diffusion" term is simply multiplying pointwise by D, i.e.
% % multiplication with a diagonal matrix on the left
% Dv = vec(A.D);
% D  = spdiags( Dv, 0, N, N );
% J  = J + D * L;
% 
% % "Flux diffusion" is more complex; first, need forward/backward
% % difference operators
% Dxb = spdiags( (1/hx) * repmat([-1,1],Nx,1), [-1 0], Nx, Nx );
% Dyb = spdiags( (1/hy) * repmat([-1,1],Ny,1), [-1 0], Ny, Ny );
% Dzb = spdiags( (1/hz) * repmat([-1,1],Nz,1), [-1 0], Nz, Nz );
% Dxf = spdiags( (1/hx) * repmat([-1,1],Nx,1), [ 0 1], Nx, Nx );
% Dyf = spdiags( (1/hy) * repmat([-1,1],Ny,1), [ 0 1], Ny, Ny );
% Dzf = spdiags( (1/hz) * repmat([-1,1],Nz,1), [ 0 1], Nz, Nz );
% 
% % Periodic boundary conditions
% [ Dxb(1,end), Dyb(1,end), Dzb(1,end), Dxf(end,1), Dyf(end,1), Dzf(end,1) ] = ...
%     deal(-1/hx, -1/hy, -1/hz, 1/hx, 1/hy, 1/hz);
% 
% % Construct full matrices via Kronecker product, and add averaged
% % forward/backward difference "flux diffusion" terms one by one (just
% % to save memory)
% Dxb = kron(Iz, kron(Iy, Dxb)); % Forward difference matrix: d/dx
% Dxf = kron(Iz, kron(Iy, Dxf)); % Forward difference matrix: d/dx
% J = J + (1/2) * ( toDiag(Dxf*Dv) * Dxb + toDiag(Dxb*Dv) * Dxf );
% clear Dxb Dxf
% 
% Dyb = kron(Iz, kron(Dyb, Ix)); % backward d/dy
% Dyf = kron(Iz, kron(Dyf, Ix)); % forward d/dy
% J = J + (1/2) * ( toDiag(Dyf*Dv) * Dyb + toDiag(Dyb*Dv) * Dyf );
% clear Dyb Dyf
% 
% Dzb = kron(kron(Dzb, Iy), Ix); % backward d/dz
% Dzf = kron(kron(Dzf, Iy), Ix); % forward d/dz
% J = J + (1/2) * ( toDiag(Dzf*Dv) * Dzb + toDiag(Dzb*Dv) * Dzf );
% clear Dzb Dzf

