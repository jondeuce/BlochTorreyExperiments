function Kernel3D = getFFTGaussianKernel(A,dt)
%GETFFTGAUSSIANKERNEL

sig = sqrt( 2 * A.D * dt );
vox = A.h(1); % isotropic; doesn't matter which
min_width = 8; % min width of gaussian (in standard deviations)
width = ceil( min_width / (vox/sig) );
unitsum = @(x) x/sum(x(:));
Gaussian1 = unitsum( exp( -0.5 * ( (-width:width).' * (vox/sig) ).^2 ) );
Gaussian2 = Gaussian1(:).';
Gaussian3 = reshape( Gaussian1, 1, 1, [] );
Gaussian2D = unitsum( Gaussian1 * Gaussian2 );
Gaussian3D = unitsum( bsxfun(@times, Gaussian2D, Gaussian3) );

% persistent Kernel3D Gaussian3D_last

% if isempty(Gaussian3D_last) || ~isequal(Gaussian3D_last,Gaussian3D)
%     Gaussian3D_last    =   Gaussian3D;
Kernel3D  =  padfastfft( Gaussian3D, A.gsize - size(Gaussian3D), true, 0 );
Kernel3D  =  fftn( ifftshift( Kernel3D ) );
% end

end
