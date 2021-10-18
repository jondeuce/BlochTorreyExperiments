function [ b ] = toeplitz_BT_multiply( n, f, Au, x )
%TOEPLITZ_BT_MULTIPLY Performs optimized matrix multiplication for toeplitz
%matrices using the fft.

N	=   prod(2*n-1);
xz	=	toeplitz_BT_pad(n,f,x);
Au	=	fft(fliplr(Au),N);
xz	=	fft(xz,N);
bz	=	Au.*xz;
bz	=	ifft(bz,N);
b	=	toeplitz_BT_reconstruct(n,f,bz);

end

% Construct random full toeplitz matrices from 3D RHS data
% Corresponds to n = [M,M,M], f = 1
%{
M	=	16;
N	=	2*M-1;
b	=	complex(randn([N,N,N]),randn([N,N,N]));
b0	=	floor(size(b)/2)+1;
A	=	zeros(M^3,M^3);
if M > 20, error( 'M too large! Take M <= 20' ); end
for ix = 1:M
    for iy = 1:M
        for iz = 1:M
            for jx = 1:M
                for jy = 1:M
                    for jz = 1:M
                        ii          =	ix+M*(iy-1+M*(iz-1));
                        jj          =	jx+M*(jy-1+M*(jz-1));
                        A(ii,jj)	=	b( b0(1)+jx-ix, b0(2)+jy-iy, b0(3)+jz-iz );
                    end
                end
            end
        end
    end
end
n	=   [M,M,M];
f	=   1;
x	=   rand(1,M^3);
Au	=   b(:).';
%}