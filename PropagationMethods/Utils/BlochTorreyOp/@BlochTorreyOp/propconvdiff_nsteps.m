function [M,Signal,Time] = propconvdiff_nsteps( A, n, dt, x0, type )
%PROPCONVDIFF_NSTEPS 

ScaleSum = prod(A.gdims)/prod(A.gsize);

Time = dt*(0:n).';
Signal = zeros(n+1,1);

M0 = double(1i);
Signal(1) = M0 * prod(A.gdims);

M = zeros('like',x0);
E2 = exp( (-dt/2) * A.Gamma );
K = getFFTGaussianKernel(A,dt);

starttime = tic;

for jj = 1:n
    looptime = tic;
    
    M = M .* E2; % decay for first half-step
    M = ifftn(fftn(M).*K); % free diffusion via convolution
    M = M .* E2; % decay for second half-step
    
    Signal(jj+1) = ScaleSum * sum(M(:));
    
    if ( 2*jj == n ) && strcmpi( type, 'SE' )
        M = conj(M);
    end
    
    str = sprintf('t = %4.1fms', 1000*dt*jj);
    display_toc_time(toc(looptime),str);
end

str = sprintf('S = %1.6e + %1.6ei', real(Signal(end)), imag(Signal(end)));
display_toc_time(toc(starttime),str);

end