function fseries = fourier_series(fcoeff,x,T)
% 
% Calculate the Fourier series expansion, given the coefficients
% 
% 
%USAGE
%-----
% fseries = fourier_series(fcoeff,x,T)
% 
% 
%INPUT
%-----
% - FCOEFF: Fourier coefficients in the form [a0 a1 ... aM b1 ... bM]
% - X     : independent variable (Nx1 vector)
% - T     : period of the function (scalar)
% 
% 
%OUTPUT
%------
% f(t) = a_0 + \sum_{m=1}^M { a_m cos(2pi m t/T) + b_m sin(2pi m t/T) }
% 
% 
% See also FOURIER_COEFF

% Guilherme Coco Beltramini (guicoco@gmail.com)
% 2011-Jul-25, 10:35 pm

N  = length(x);
M  = (length(fcoeff)-1)/2;
w0 = 2*pi/T;
series = zeros(N,2*M+1);
series(:,1) = 1;
for m=2:M+1
    series(:,m)   = cos(w0*(m-1)*x);
    series(:,m+M) = sin(w0*(m-1)*x);
end
fseries = series*reshape(fcoeff,2*M+1,1);