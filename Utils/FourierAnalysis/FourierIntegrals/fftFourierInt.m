function [ I, w ] = fftFourierInt( h, a, b, M, order, isIFFT, dim, loc )
%FFTFOURIERINT Approximates fourier integrals using the fft:
% 
%   I	=   int( h(x) * exp(-i*w*x), a, b )
% 
% where:
%   -w are the M equally spaced frequencies defined by:
%       w(m) = m * (2*pi)/(b-a), for m = -floor(M/2):ceil(M/2)-1
%   -a, b are the left, right endpoints of the integral
%   -h(x) is evaluated at the M+1 equally spaced points x defined by:
%       x = linspace(a,b,M+1)

%==========================================================================
% Parse Inputs
%==========================================================================
isfunc	=   isa( h, 'function_handle' );

if nargin < 4 && isfunc
    error( 'Must provide number of points M if h is a function handle' );
end

if nargin < 5 || isempty(order)
    order	=   'trap';
end

if nargin < 6 || isempty(isIFFT)
    isIFFT	=   false;
end

if nargin < 7 || isempty(dim)
    if isfunc
        dim     =   1;
    else
        % dim is the (first) longest dimension
        dim     =   find( size(h) == length(h), true, 'first' );
    end
end

if nargin < 8 || isempty(loc)
    loc     =	'endpoints';
end
if strcmpi( loc, 'interior' )
    dx      =	(b-a)/(M+1);
    [a,b]	=   deal( a+dx/2, b-dx/2 );
end

if isfunc
    t	=   reshapeVec( linspace(a,b,M+1), dim );
    h	=   reshapeVec( h(t), dim );
else
    if isempty(M)
        M	=   size(h,dim)-1;
    end
end

%==========================================================================
% Evaluate Integrals
%==========================================================================
switch upper(order)
    case 'CUBIC'
        [I,w]	=   fftFourierInt_cubic( h, a, b, M, isIFFT, dim );
    case 'GAUSSIAN'
        warning( '''gaussian'' is not implemented. Using default ''trap'' instead' );
%         [I,w]	=   fftFourierInt_gaussian_periodic3D( h, a, b, M, isIFFT, dim );
        [I,w]	=   fftFourierInt_trap( h, a, b, M, isIFFT, dim );
    otherwise %default is 'TRAP'
        [I,w]	=   fftFourierInt_trap( h, a, b, M, isIFFT, dim );
end

end

function I = handle_constant_h(h,wn,dim)

% Expand to length(wn) along dim
Isize       =   size(h);
Isize(dim)	=   length(wn);
I           =   zeros(Isize,'like',h);

% Set DC term to h and return
I           =   setsliceND( I, find(wn==0), dim, h );

end

function [I,wn] = fftFourierInt_trap( h, a, b, M, isIFFT, dim )

sgn	=   -1;
if isIFFT
    sgn	=   1;
end

N	=   M;
D	=   (b-a)/M;
n	=   reshapeVec( -floor(N/2):ceil(N/2)-1, dim );
wn	=   2*pi*n/(N*D);
th	=   wn*D;

if ( size(h,dim) == 1 ) && ( M > 0 )
    % Constant function
    I	=	handle_constant_h(h,wn,dim);
    return
end

L	=   size(h,dim);
H	=   sliceND(h,1:L-1,dim);
if isIFFT
    H	=   ifft( N * H, [], dim );
else
    H	=   fft( H, [], dim );
end
H	=   fftshift( H, dim );

W	=   W_trap(sgn*th);
a0	=   alpha_trap(sgn*th);
A	=  -a0;

% 1D Version generalized to any dimension:
% I	=	D*exp(1i*wn*a) * ( W*H + a0*h(1) + exp(1i*wn*(b-a))*A*h(end) )

h0	=   sliceND(h,1,dim); % left boundary
hE	=   sliceND(h,L,dim); % right boundary
I	=   bsxfun( @times,	W,	H );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a0, h0 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, exp(sgn*1i*wn*(b-a)).*A, hE ) );
I	=   bsxfun( @times,	I,	D*exp(sgn*1i*wn*a) );

% Testing
%{
I2	=	zeros(N,1);
t	=   linspace(a,b,M+1).';

% Random 'even' smooth periodic data 
% h = rand(2*(M+1)+1,1); h(M+3:end) = h(M+1:-1:1);
% h = conv(h,exp(-(-36:36).^2/(2*(6)^2))/sum(exp(-(-36:36).^2/(2*(6)^2))),'same');
% h = h(M/2+2:M/2+M+2);

% Random 'odd' smooth periodic data 
% h = rand(2*M,1); h(M+1:end) = h(1:M);
% h = conv(h,exp(-(-36:36).^2/(2*(6)^2))/sum(exp(-(-36:36).^2/(2*(6)^2))),'same');
% h = h(M/2+1:3*M/2+1);

% add discontinuity
% h = h + (t>-rand*pi/2 & t<rand*pi/2);

% linear interpolator
fp	=   @(x) reshape( interp1(t,h,x(:),'linear'), size(x) );

% plot
figure, plot(t,h,'b.-'), hold on
plot(t,fp(t),'r--')
xlim([a,b]);

for ii = 1:length(n)
    g       =	@(f,t) reshape(f(t),size(t)).*exp(sgn*1i*wn(ii)*t);
    I2(ii)	=	integral( @(t)g(fp,t), a, b, 'abstol', 1e-14, 'reltol', 1e-14 );
end

x = linspace(a,b,M+1).';
y = zeros(size(x));
for ii = 1:length(n)
    y = y + exp(1i*wn(ii)*x)*I(ii)/(b-a);
end
plot(x,y,'k--');

%}

end

function W = W_trap(th)

tol     =   1e-14;
th2     =   th.*th;

b_0th	=   ( th2 < tol * 12 );
b_2nd	=   ( th2 < sqrt( tol * 360 ) )	& ~b_0th;
b_4th	=   ( th2 < (tol * 20160).^(1/3) ) & ~(b_0th | b_2nd);
b_exact	=  ~( b_0th | b_2nd | b_4th );

W           =   zeros(size(th));
W(b_0th)	=   1;
W(b_2nd)	=   1 - (1/12)*th2(b_2nd);
W(b_4th)	=   1 + th2(b_4th).*(th2(b_4th)/360 - 1/12);
W(b_exact)	=   (2./th2(b_exact)).*(1-cos(th(b_exact)));

% Testing
%{
f_W     =   @(th) (2./(th.^2)).*(1-cos(th));
f_logW	=	@(th) log(2) + log(1-cos(th)) - 2*log(th);
th      =	logspace(-14,log10(pi),100).';

close all force
figure
h = loglog(th,[abs(f_logW(th)), abs(log(abs(W)))]);
legend(h,'Closed Form','Taylor Series');
title('abs(log(abs(W(\theta))))');
%}

end

function a0 = alpha_trap(th)

tol     =   1e-14;
th2     =   th.*th;

b_0th	=   ( th2 < tol * 24 );
b_2nd	=   ( th2 < sqrt( tol * 720 ) )	& ~b_0th;
b_4th	=   ( th2 < (tol * 40320).^(1/3) ) & ~(b_0th | b_2nd);
b_exact	=  ~( b_0th | b_2nd | b_4th );

a0          =   zeros(size(th));
a0(b_0th)	=   complex(   -0.5, th(b_0th)*(1/6) );
a0(b_2nd)	=   complex(   -0.5 + (1/24)*th2(b_2nd), ...
                            th(b_2nd).*(1/6 - (1/120)*th2(b_2nd))	);
a0(b_4th)	=   complex(   -0.5 - th2(b_4th).*(th2(b_4th)/720 - 1/24), ...
                            th(b_4th).*(1/6+th2(b_4th).*(th2(b_4th)/5040 - 1/120))  );
a0(b_exact)	=   complex(	(-1./th2(b_exact)).*(1-cos(th(b_exact))), ...
                            ( 1./th2(b_exact)).*(th(b_exact)-sin(th(b_exact)))	);

% Testing
%{
f_loga0real	=	@(th) log(1-cos(th)) - 2*log(th);
f_loga0imag	=	@(th) log(th-sin(th)) - 2*log(th);
th          =	logspace(-14,log10(pi),100).';

close all force
figure
h = loglog(th,[f_loga0real(th), log(abs(real(a0)))]);
legend(h,'Closed Form','Taylor Series');
title('log(abs(real(\alpha_0(\theta))))');

figure
h = loglog(th,[f_loga0imag(th), log(abs(imag(a0)))]);
legend(h,'Closed Form','Taylor Series');
title('log(abs(imag(\alpha_0(\theta))))');
%}
                        
end

function [I,wn] = fftFourierInt_cubic( h, a, b, M, isIFFT, dim )

sgn	=   -1;
if isIFFT
    sgn	=   1;
end

N	=   M;
D	=   (b-a)/M;
n	=   reshapeVec( -floor(N/2):ceil(N/2)-1, dim );
wn	=   2*pi*n/(N*D);
th	=   wn*D;

if ( size(h,dim) == 1 ) && ( M > 0 )
    % Constant function
    I	=	handle_constant_h(h,wn,dim);
    return
end

L	=   size(h,dim);
H	=   sliceND(h,1:L-1,dim);
if isIFFT
    H	=   ifft( N * H, [], dim );
else
    H	=   fft( H, [], dim );
end
H	=   fftshift( H, dim );

W               =   W_cubic(sgn*th);
[a0,a1,a2,a3,A]	=   alpha_cubic(sgn*th);

% 1D Version generalized to any dimension:
% I	=	D*exp(sgn*1i*wn*a)   .* (	W.*H + a0*h(1) + a1*h(2) + a2*h(3) + a3*h(4) + ...
%       exp(sgn*1i*wn*(b-a)) .* (	A*h(end) + conj(a1)*h(end-1) + conj(a2)*h(end-2) + conj(a3)*h(end-3) ) );

h0	=   sliceND(h,1,  dim); % left boundary
h1	=   sliceND(h,2,  dim); % one inside left boundary
h2	=   sliceND(h,3,  dim); % two inside left boundary
h3	=   sliceND(h,4,  dim); % three inside left boundary
hE3	=   sliceND(h,L-3,dim); % three inside right boundary
hE2	=   sliceND(h,L-2,dim); % two inside right boundary
hE1	=   sliceND(h,L-1,dim); % one inside right boundary
hE0	=   sliceND(h,L,  dim); % right boundary

I	=   bsxfun( @times,	A,	hE0 );
I	=   I + bsxfun( @times, conj(a1), hE1 );
I	=   I + bsxfun( @times, conj(a2), hE2 );
I	=   I + bsxfun( @times, conj(a3), hE3 );
I	=   bsxfun( @times,	I,	exp( (sgn*1i*(b-a)) * wn ) );

I	=   bsxfun( @plus,	I,	bsxfun( @times,	W,	H  ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a0, h0 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a1, h1 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a2, h2 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a3, h3 ) );
I	=   bsxfun( @times,	I,	D * exp( (sgn*1i*a) * wn ) );


% Testing
%{
I2	=	zeros(N,1);
t	=   linspace(a,b,M+1).';

% Random 'even' smooth periodic data 
% h = rand(2*(M+1)+1,1); h(M+3:end) = h(M+1:-1:1);
% h = conv(h,exp(-(-36:36).^2/(2*(6)^2))/sum(exp(-(-36:36).^2/(2*(6)^2))),'same');
% h = h(M/2+2:M/2+M+2);

% Random 'odd' smooth periodic data 
% h = rand(2*M,1); h(M+1:end) = h(1:M);
% h = conv(h,exp(-(-36:36).^2/(2*(6)^2))/sum(exp(-(-36:36).^2/(2*(6)^2))),'same');
% h = h(M/2+1:3*M/2+1);

% add discontinuity
% h = h + (t>-rand*pi/2 & t<rand*pi/2);

% Cubic spline interpolator
pp	=	spline(t,h);
fp	=   @(x) ppval(pp,x);

% plot
figure, plot(t,h,'b.-'), hold on
fplot(fp,[a,b],'r--')

for ii = 1:length(n)
    g       =	@(f,t) reshape(f(t),size(t)).*exp(sgn*1i*wn(ii)*t);
    I2(ii)	=	integral( @(t)g(fp,t), a, b, 'abstol', 1e-12, 'reltol', 1e-6 );
end
%}

end

function W = W_cubic(th)

tol     =   1e-10;
th2     =   th.*th;

b_0th	=   ( th2 < tol * 12 );
b_4th	=   ( th2 < (tol * 15120/23).^(1/3) ) & ~(b_0th);
b_exact	=  ~( b_0th | b_4th );

W           =   zeros(size(th));
W(b_0th)	=   1;
W(b_4th)	=   1 - (11/720)*(th2(b_4th).*th2(b_4th));
W(b_exact)	=   ((6+th2(b_exact))./(3*th2(b_exact).*th2(b_exact))) ...
                .* (3-4*cos(th(b_exact))+cos(2*th(b_exact)));

% Testing
%{
f_W     =   @(th) ((6+th.^2)./(3*th.^4)).*(3-4*cos(th)+cos(2*th));
f_logW	=	@(th) log(6+th.^2) - log(3*th.^4) + log(3-4*cos(th)+cos(2*th));
th      =	logspace(-14,log10(pi),1000).';

close all force
figure
h = loglog(th,[abs(f_logW(th)), abs(log(abs(W)))]);
legend(h,'Closed Form','Taylor Series');
title('abs(log(abs(W(\theta))))');
%}

end

function [a0,a1,a2,a3,A] = alpha_cubic(th)

tol     =   1e-14;
th2     =   th.*th;
th4     =   th2.*th2;
th6     =   th4.*th2;

a0r	=   (-2.0./3.0)+th2./45.0+(103.0./15120.0).*th4-(169.0./226800.0).*th6;
a1r	=   (7.0./24.0)-(7.0./180.0).*th2+(5.0./3456.0).*th4-(7.0./259200.0).*th6;
a2r	=   (-1.0./6.0)+th2./45.0-(5.0./6048.0).*th4+th6./64800.0;
a3r	=   (1.0./24.0)-th2./180.0+(5.0./24192.0).*th4-th6./259200.0;
a0i	=   th.*(2.0./45.0+(2.0./105.0).*th2-(8.0./2835.0).*th4+(86.0./467775.0).*th6);
a1i	=   th.*(7.0./72.0-th2./168.0+(11.0./72576.0).*th4-(13.0./5987520.0).*th6);
a2i	=   th.*(-7.0./90.0+th2./210.0-(11.0./90720.0).*th4+(13.0./7484400.0).*th6);
a3i	=   th.*(7.0./360.0-th2./840.0+(11.0./362880.0).*th4-(13.0./29937600.0).*th6);
Ar	=   (1.0./3.0)+(1.0./45.0).*th2-(8.0/945.0).*th4+(11.0/14175.0).*th6;

b0      =   ( abs(th) < (tol * 226800./169).^(1./6) );
a0      =   zeros(size(th));
a0(b0)	=	complex( a0r(b0), a0i(b0) );

b1	=   ( abs(th) < (tol * 259200./7).^(1./6) );
a1      =   zeros(size(th));
a1(b1)	=	complex( a1r(b1), a1i(b1) );

b2	=   ( abs(th) < (tol * 259200./7).^(1./6) );
a2      =   zeros(size(th));
a2(b2)	=	complex( a2r(b2), a2i(b2) );

b3	=   ( abs(th) < (tol * 259200).^(1./6) );
a3      =   zeros(size(th));
a3(b3)	=	complex( a3r(b3), a3i(b3) );

bA	=   ( abs(th) < (tol * 14175/11).^(1./6) );
A       =   zeros(size(th));
A(bA)	=   complex( Ar(bA), -imag(a0(bA)) );

cth=cos(th);
sth=sin(th);
ctth=cth.*cth-sth.*sth;
stth=2.0e0.*sth.*cth;
tmth2=3.0e0-th2;
spth2=6.0e0+th2;
sth4i=1.0./(6.0e0.*th4);
tth4i=2.0e0.*sth4i;
a0r=sth4i.*(-42.0e0+5.0e0.*th2+spth2.*(8.0e0.*cth-ctth));
a0i=sth4i.*(th.*(-12.0e0+6.0e0.*th2)+spth2.*stth);
a1r=sth4i.*(14.0e0.*tmth2-7.0e0.*spth2.*cth);
a1i=sth4i.*(30.0e0.*th-5.0e0.*spth2.*sth);
a2r=tth4i.*(-4.0e0.*tmth2+2.0e0.*spth2.*cth);
a2i=tth4i.*(-12.0e0.*th+2.0e0.*spth2.*sth);
a3r=sth4i.*(2.0e0.*tmth2-spth2.*cth);
a3i=sth4i.*(6.0e0.*th-spth2.*sth);
Ar =sth4i.*((-6.0+11.0.*th2)+(6.0+th2).*ctth);

a0(~b0)	=	complex( a0r(~b0), a0i(~b0) );
a1(~b1)	=	complex( a1r(~b1), a1i(~b1) );
a2(~b2)	=	complex( a2r(~b2), a2i(~b2) );
a3(~b3)	=	complex( a3r(~b3), a3i(~b3) );
A(~bA)	=   complex( Ar( ~bA), -imag(a0(~bA)) );

% Testing
%{
th          =	logspace(-14,log10(pi),1000).';

close all force
figure
h = loglog(th,log(abs(real(a0))));
legend(h,'Taylor Series');
title('log(abs(real(\alpha_0(\theta))))');
figure
h = loglog(th,log(abs(real(a1))));
legend(h,'Taylor Series');
title('log(abs(real(\alpha_1(\theta))))');
figure
h = loglog(th,log(abs(real(a2))));
legend(h,'Taylor Series');
title('log(abs(real(\alpha_2(\theta))))');
figure
h = loglog(th,log(abs(real(a3))));
legend(h,'Taylor Series');
title('log(abs(real(\alpha_3(\theta))))');
figure
h = loglog(th,log(abs(real(A))));
legend(h,'Taylor Series');
title('log(abs(real(A(\theta))))');
%}

end

function [I,wn] = fftFourierInt_gaussian_periodic3D( h, a, b, M, isIFFT, dim )

sgn	=   -1;
if isIFFT
    sgn	=   1;
end

N	=   M;
D	=   (b-a)/M;
n	=   reshapeVec( -floor(N/2):ceil(N/2)-1, dim );
wn	=   2*pi*n/(N*D);
th	=   wn*D;

if ( size(h,dim) == 1 ) && ( M > 0 )
    % Constant function
    I	=	handle_constant_h(h,wn,dim);
    return
end

L	=   size(h,dim);
H	=   sliceND(h,1:L-1,dim);

if isIFFT
    H	=   ifft( N * H, [], dim );
else
    H	=   fft( H, [], dim );
end
H	=   fftshift( H, dim );

% 1D Version generalized to any dimension:
% I	=	D*exp(sgn*1i*wn*a)   .* (	W.*H + a0*h(1) + a1*h(2) + a2*h(3) + a3*h(4) + ...
%       exp(sgn*1i*wn*(b-a)) .* (	A*h(end) + conj(a1)*h(end-1) + conj(a2)*h(end-2) + conj(a3)*h(end-3) ) );

W                   =	exp(-0.5*(th.*th));
[a0,a1,a2,a3,a4,a5]	=	alpha_gaussian(th);

h0	=   sliceND(h,1,  dim); % left boundary
h1	=   sliceND(h,2,  dim); % one inside left boundary
h2	=   sliceND(h,3,  dim); % two inside left boundary
h3	=   sliceND(h,4,  dim); % three inside left boundary
h4	=   sliceND(h,5,  dim); % four inside left boundary
h5	=   sliceND(h,6,  dim); % five inside left boundary
hE5	=   sliceND(h,L-5,dim); % five inside right boundary
hE4	=   sliceND(h,L-4,dim); % four inside right boundary
hE3	=   sliceND(h,L-3,dim); % three inside right boundary
hE2	=   sliceND(h,L-2,dim); % two inside right boundary
hE1	=   sliceND(h,L-1,dim); % one inside right boundary
hE0	=   sliceND(h,L,  dim); % right boundary

I	=   bsxfun( @times,	W+conj(a0), hE0 );
I	=   I + bsxfun( @times, conj(a1), hE1 );
I	=   I + bsxfun( @times, conj(a2), hE2 );
I	=   I + bsxfun( @times, conj(a3), hE3 );
I	=   I + bsxfun( @times, conj(a4), hE4 );
I	=   I + bsxfun( @times, conj(a5), hE5 );
I	=   bsxfun( @times,	I,	exp( (sgn*1i*(b-a)) * wn ) );

I	=   bsxfun( @plus,	I,	bsxfun( @times,	W,	H  ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a0, h0 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a1, h1 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a2, h2 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a3, h3 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a4, h4 ) );
I	=   bsxfun( @plus,	I,	bsxfun( @times, a5, h5 ) );
I	=   bsxfun( @times,	I,	D * exp( (sgn*1i*a) * wn ) );


%{
% old version (works for dim 1 only)
N	=   2^nextpow2(M+1);
D	=   (b-a)/M;
n	=   (-floor(N/2):ceil(N/2)-1).';
wn	=   2*pi*n/(N*D);
th	=   wn*D;

hp	=   [h;zeros(N-M-1,1)];
H	=   fftshift(fft(hp));
W	=   exp(-0.5*(th.*th));
I	=   D*exp(sgn*1i*wn*a) .* W .* H;
%}

% Testing
%{
I2	=	zeros(N,1);
t	=   linspace(a,b,M+1).';

% Periodic gaussian 'interpolator'
fg	=   @(x,t,s) exp(-(0.5/s^2)*bsxfun(@minus,x(:)/D,t(:).'/D).^2)/(sqrt(2*pi)*s);
hp	=   [h(end-6:end-1); h(1:end-1); h(1:6)];
tp	=   a + D*(-6:M+6-1);
fh	=   @(x) sum( bsxfun(@times, hp(:).', fg(x,tp,1.0)), 2 );

% plot
figure, plot(t,h,'b.-'), hold on
fplot(fh,[a,b],'g--')

for ii = 1:length(n)
    g       =	@(f,t) reshape(f(t),size(t)).*exp(sgn*1i*wn(ii)*t);
    I2(ii)	=	integral( @(t)g(fh,t), a, b, 'abstol', 1e-14, 'reltol', 1e-6 );
end

x = linspace(a,b,N).';
y = zeros(size(x));
for ii = 1:length(n)
    y = y + exp(1i*wn(ii)*x)*I(ii)/(b-a)/4;
end
plot(x,y,'k--');

%}

end

function [a0,a1,a2,a3,a4,a5] = alpha_gaussian(th)

% We have that:
%   phi(s)      :=	exp(-s^2/2)/sqrt(2*pi)
% 
% We define (for 0 <= j <= 5):
%   phi_j(s)	:=	{	phi(s)/N(j),	s >= -j
%                   {   0,              s <  -j
%   N(j)        :=  int( phi(s), -j, inf )
%                =	1/2 * ( 1 + erf(j/sqrt(2)) )
% 
% The difference is then:
%   psi_j(s)	:=	phi_j(s) - phi(s)
%                =	{   phi(s-j)/N(j) - phi(s), s >= 0
%                =  {  -phi(s),                 s <  0
% Then,
%   a_j(t)	:=	int( exp(i*t*s) * psi_j(s), -inf, inf )
%            =  int( exp(i*t*s) * phi(s-j), 0, inf )/N(j) - int( exp(i*t*s) * phi(s), -inf, inf )
%            =  exp( -t^2/2+i*j*t ) * ( 1+erf((j+i*t)/sqrt(2)) ) / (2*N(j)) - exp( -t^2/2 )
%            =  exp( -t^2/2 ) * ( 0.5 .* exp( i*j*t ) * (1 + erf((j+i*t)/sqrt(2)))./N(j) - 1 )
% 
% Even symmetry implies:
%   phi_(M-j)(s) =  phi_j(-s)
%   a_(M-j)(t)   =  exp(i*w*(b-a)) * conj(a_j(t))

Nj	=   @(j)	0.5 * ( 1.0 + Faddeeva_erf((0.5*sqrt(2.0)).*j) );
aj	=   @(j,t)	exp( -0.5.*t.*t ) .* ...
    ( exp( 1i.*j.*t ) .* (1+Faddeeva_erf(0.5.*sqrt(2.0).*(j+1i.*t)))./(2.*Nj(j)) - 1.0 );

[a0,a1,a2,a3,a4,a5]	=	deal( ...
    aj(0,th), aj(1,th), aj(2,th), aj(3,th), aj(4,th), aj(5,th) );

end

function y = gauss_interp(a,b)

h       =	humps(linspace(0,1,101).') .* sin(linspace(0,pi,101).');

phi     =	@(s)	exp(-0.5*s.^2)/sqrt(2*pi);
% Nj      =	@(j)	0.5 * ( 1.0 + Faddeeva_erf((0.5*sqrt(2.0)).*j) );
% psi_j	=	@(j,s)	phi(s)./Nj(j).*(s>=-j) - phi(s);
Cj      =   @(j)	1 + phi(j).*(1-sum(phi(0:5)))./sum(phi(0:5).^2);
psi_j	=	@(j,s)	(Cj(j)-1).*phi(s);

h	=   h(:);
M	=   length(h)-1;
D	=   (b-a)/M;
t	=   linspace(a,b,10*M+1).';
s	=   t./D;
sj	=   a/D + (0:M).';

% phi(s) terms
y	=   zeros(size(s));
for jj = 0:M
    y	=   y + h(jj+1) * phi(s-sj(jj+1));
end

% psi_j(s) terms
for jj = 0:5
    y	=   y + h(jj+1)	  * psi_j(jj,s-sj(jj+1)) + ...
                h(M-jj+1) * psi_j(jj,-(s-sj(M-jj+1)));
end

end






















