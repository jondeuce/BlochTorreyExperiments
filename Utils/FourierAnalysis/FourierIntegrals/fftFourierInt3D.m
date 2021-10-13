function [ I, wx, wy, wz ] = fftFourierInt3D( h, xb, yb, zb, Mx, My, Mz, order, isIFFT, loc )
%FFTFOURIERINT3D Approximates 3D fourier integrals using the fft

%==========================================================================
% Parse Inputs
%==========================================================================
isfunc	=   isa( h, 'function_handle' );
if nargin < 7 && isfunc
    error( 'Must provide number of points M if h is a function handle' );
end

if nargin < 8
    order	=   'trap';
end

if nargin < 9
    isIFFT	=   false;
end

if isfunc
    tx	=   linspace(xb(1),xb(2),Mx+1).';
    ty	=   linspace(yb(1),yb(2),My+1).';
    tz	=   linspace(zb(1),zb(2),Mz+1).';
    h	=   h(tx,ty,tz);
else
    if isempty(Mx) || isempty(My) || isempty(Mz)
        [Mx,My,Mz]	=   size(h);
        [Mx,My,Mz]	=	deal(Mx-1,My-1,Mz-1);
    end
end

%==========================================================================
% Evaluate Integrals
%==========================================================================
% [ I, w ]	=	fftFourierInt( h, a, b, M, order, isIFFT, dim )

[ I, wx ]	=   fftFourierInt( h, xb(1), xb(2), Mx, order, isIFFT, 1, loc );
[ I, wy ]	=   fftFourierInt( I, yb(1), yb(2), My, order, isIFFT, 2, loc );
[ I, wz ]	=   fftFourierInt( I, zb(1), zb(2), Mz, order, isIFFT, 3, loc );

end




