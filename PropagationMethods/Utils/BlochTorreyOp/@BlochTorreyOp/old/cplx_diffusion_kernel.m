function [ h, I ] = cplx_diffusion_kernel( x, t, c )
%CPLX_DIFFUSION_KERNEL Fundamental solution h(x,t) for the linear diffusion
%equation with complex diffusion coefficient c with real(c) > 0.
% 
%   u_t = c * u_xx
% 
%   In higher dimensions, the fundamental solution is the product of lower
%   dimensional solutions along each spatial dimension, i.e. in 3D
%       h_3D(x,y,z,t) = h(x,t) * h(y,t) * h(z,t)
% 
%   N.B. The integral of the kernel in 1D is given by
%       I = 1/sqrt(1-1i*a)
%   where
%       a = sigma^2 * sin(theta) / (2*r*t)
%       r = abs(c), theta = angle(c)
%       sigma = sqrt(2*t*r/cos(theta))
% 
%	1.	Gilboa G, Zeevi YY, Sochen NA. Complex diffusion processes for
%       image filtering. In: Scale-Space. Springer, 2001, pp. 299–307.

r = abs(c);
theta = angle(c);
sigma = sqrt(2*t*r/cos(theta));

C = 1/(sqrt(2*pi)*sigma);
D = 1/(2*sigma^2);
E = sin(theta)/(4*t*r);
F = -D + 1i*E;

h = C .* exp( F .* (x.^2) );

if nargout > 1
    a = sigma^2 * sin(theta) / (2*r*t);
    I = 1/sqrt(1-1i*a);
end

end

