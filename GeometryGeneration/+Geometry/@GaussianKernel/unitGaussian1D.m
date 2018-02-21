function [g,I] = unitGaussian1D(vox, sig, dim)
%UNITGAUSSIAN1D Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3; dim = 1; end

if sig/vox <= 10*eps(1)
    g = 1; I = 1;
    return
end

if real(sig^2) <= 0
    error('Real part of standard deviation squared must be strictly positive.');
end

if isreal(sig)
    [g,I] = realUnitGaussian1D(vox,sig);
else
    [g,I] = cplxUnitGaussian1D(vox,sig);
end

if dim == 2
    g = g(:).';
elseif dim == 3
    g = reshape(g,1,1,[]);
end

end

function [g,I] = realUnitGaussian1D(vox, sig)

% 8-sigma ensures lowest value is ~machine precision
min_width = 8; % min width of gaussian (in UNITLESS standard deviations)
width = ceil( min_width / (vox/sig) );

C = 1/(sqrt(2*pi) * (sig/vox));
g = C .* exp( -0.5 * ( (-width:width).' * (vox/sig) ).^2 );

% Normalizing to 1; kernel should sum to one to preserve the limit
% g -> delta(x) as sig -> 0
I = 1;
g = g * (I/sum(g(:)));

end

function [h,I] = cplxUnitGaussian1D(vox, sig)
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

% Complex diffusion coefficient in terms of complex standard deviation:
t = 1;
c = sig^2/(2*t); % sig = sqrt(2*c*t)

% Convert to standard form according to referenced paper above; it is this
% sigma which determines the width of the kernel
r = abs(c);
theta = angle(c);
sigma = sqrt(2*t*r/cos(theta));

% 8-sigma ensures lowest value is ~machine precision
min_width = 8; % min width of gaussian (in UNITLESS standard deviations)
width = ceil( min_width / (vox/sigma) );

% Calculate complex gaussian kernel
x2 = ((-width:width).' * vox).^2;
alpha = x2 * sin(theta)/(4*t*r);

g = exp(-x2/(2*sigma^2)) / sqrt(2*pi*(sigma/vox)^2);
h = g .* exp(1i*alpha);

% Normalizing to 1; kernel should sum to one to preserve the limit
% h -> delta(x) as sig -> 0
I = 1;
h = h * (I/sum(h(:)));

end









