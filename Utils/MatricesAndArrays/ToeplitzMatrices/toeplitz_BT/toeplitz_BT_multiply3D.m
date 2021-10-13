function [ b ] = toeplitz_BT_multiply3D( n, f, Au, x )
%TOEPLITZ_BT_MULTIPLY3D Performs optimized matrix multiplication for 
% toeplitz matrices using the fft.

% Pad x with zeros
if f == 1
    xz                          =	zeros(2*n-1);
    xz(1:n(1),1:n(2),1:n(3))	=   x;
else
    xz	=	toeplitz_BT_pad(n,f,x(:).').';
end

% Convolve toeplitz matrix Au with x
N	=   prod(2*n-1);
Au	=   flipud(Au(:));
Au	=	fft(Au,N);
xz	=	fft(xz(:),N);
b	=	Au.*xz;
b	=	ifft(b,N);
b	=   reshape(b,2*n-1);

% Unpad b and return
if f == 1
    b	=   ifftshift(b);
    b	=   b(1:n(1),1:n(2),1:n(3));
else
    b	=   toeplitz_BT_reconstruct(n,f,b(:).').';
end

end

% Construct random full toeplitz matrices from 3D RHS data
% Corresponds to n = [M,M,M], f = 1
%{
M	=	[13,14,16];
N	=	2*M-1;
b	=	complex(randn(N),randn(N));
D	=   5 * diag(sort(complex(randn(prod(M),1),randn(prod(M),1))));
% b	=   zeros(N);
% b(floor(7*N(1)/16):ceil(9*N(1)/16),floor(7*N(2)/16):ceil(9*N(2)/16),floor(7*N(3)/16):ceil(9*N(3)/16)) = 1;
% mb	=   false(N);
b0	=	floor(size(b)/2)+1;
A	=	zeros(prod(M),prod(M));
if any(M > 20), error( 'M too large! Take M <= 20' ); end
for ix = 1:M(1)
    for iy = 1:M(2)
        for iz = 1:M(3)
            for jx = 1:M(1)
                for jy = 1:M(2)
                    for jz = 1:M(3)
                        ii          =	ix+M(1)*(iy-1+M(2)*(iz-1));
                        jj          =	jx+M(1)*(jy-1+M(2)*(jz-1));
                        A(ii,jj)	=	b( b0(1)+jx-ix, b0(2)+jy-iy, b0(3)+jz-iz );
%                         mb( b0(1)+jx-ix, b0(2)+jy-iy, b0(3)+jz-iz ) =	true;
                    end
                end
            end
        end
    end
end
if ~isempty(D), A = A + D; end
fe	=	@(x,y) max(abs(x(:)-y(:)))./max(abs([x(:);y(:)]));
% n	=   M;
% f	=   1;
x	=   rand(M);
% Au	=   b;
T	=	Toeplitz3D(b,[],D)
fe(A*x(:),T*x(:))
%}