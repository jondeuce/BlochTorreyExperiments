function [ S, u ] = heatLikeEqn1D( D, f, u0, Tmax )
%HEATLIKEEQN1D Solves the 1D generalized heat-like equation on the domain
% given by {-pi<=x<=pi, 0<=t<=Tmax}. The equation being solved is:
%	u_t = D*u_xx - f(x)*u

% Tmax is for convergence testing
if nargin < 4, Tmax = 1.0; end

% Spectral method
[S,b,T0,A]	=	S_func_spectral_iter(D,f,u0,Tmax);
[u,b,T0,A]	=	u_func_spectral_iter(D,f,u0,Tmax);

end

%==========================================================================
% Spectral method for signal
%==========================================================================

function [b,T0,A] = iter_spectral(D,f,u0,b,T0,N)

inv_2pi     =   1/(2*pi);
sqrt_2pi	=	sqrt(2*pi);
sqrtinv_2pi	=	1/sqrt_2pi;

M0	=   length(b);
N0	=   length(T0);
n0	=   floor(N0/2);
M	=   2*N-1;
n	=   floor(N/2);
% xx	=   linspace(-pi,pi,128).';

% [b,wb]	=	fftFourierInt( f(xx), -pi, pi, M, 'trap', true );
b	=   [ zeros(N-N0,1); b(:); zeros(N-N0,1) ];
for L = N0:N-1
    iHigh	=   L + N;
    iLow	=   M - iHigh + 1;
    
    b(iHigh)	=   inv_2pi * integral( @(x) f(x) .* exp( ( 1i*L) * x ), -pi, pi, 'abstol', 1e-6, 'reltol', 1e-6 );
    b(iLow)     =   inv_2pi * integral( @(x) f(x) .* exp( (-1i*L) * x ), -pi, pi, 'abstol', 1e-6, 'reltol', 1e-6 );
end

% [T0,wT]	=	fftFourierInt( u0(xx), -pi, pi, N, 'trap', false );
T0	=   [ zeros(n-n0,1); T0(:); zeros(n-n0,1) ];
for L = n0+1:n
	iHigh	=   L + n + 1;
    iLow	=   N - iHigh + 1;
    
    T0(iHigh)	=   sqrtinv_2pi * integral( @(x) u0(x) .* exp( (-1i*L) * x ), -pi, pi, 'abstol', 1e-6, 'reltol', 1e-6 );
    T0(iLow)	=   sqrtinv_2pi * integral( @(x) u0(x) .* exp( ( 1i*L) * x ), -pi, pi, 'abstol', 1e-6, 'reltol', 1e-6 );
end

if nargout > 2
    A	=	make_A(b,D);
%     A	=	-b(N)*eye(N) - diag(D*(-n:n).^2);
%     for L = 1:N-1
%         iHigh	=   L + N;
%         iLow	=   M - iHigh + 1;
%         A	=	A - diag( b(iHigh) * ones(N-L,1),  L );
%         A	=	A - diag( b(iLow)  * ones(N-L,1), -L );
%     end
end

end

function A = make_A(b,D)

M	=   length(b);
N	=   ceil(M/2);
n	=   floor(N/2);
k	=   ~mod(M,2);

A	=	-b(N+k)*eye(N) - diag(D*(-n:n-~mod(N,2)).^2);
for L = 1:N-1
    iHigh	=   L + N + k;
    iLow	=   M - iHigh + 1 + k;
    A	=	A - diag( b(iHigh) * ones(N-L,1),  L );
    A	=	A - diag( b(iLow)  * ones(N-L,1), -L );
end

end

function u = u_func_spectral(x,t,A,T0)

x	=   x(:);
t	=   t(:).';
N	=   length(T0);
n	=   -floor(N/2):ceil(N/2)-1;
sqrt_2pi	=   sqrt(2*pi);
isqrt_2pi	=   1/sqrt_2pi;

u           =   zeros(numel(x),numel(t));
for jj = 1:numel(t)
%     T       =   expm(A*t(jj)) * T0;
    T       =   expmv(t(jj),A,T0);
    X       =   isqrt_2pi * exp( 1i * bsxfun(@times,x,n) );
    u(:,jj)	=   sum( bsxfun(@times, T.', X), 2 );
end

end

function [u,b,T0,A] = u_func_spectral_iter(D,f,u0,Tmax)

Nmax	=   128;
NumKeep	=   32/2;
N0      =   floor(Nmax/2);
tidx	=	N0+1-NumKeep:N0+1+NumKeep;
bidx	=   N0+1-(2*NumKeep):N0+1+(2*NumKeep);

xx      =   linspace(-pi,pi,Nmax+1);
[b, wb]	=	fftFourierInt( f(xx),  -pi, pi, Nmax, 'cubic', true  );
[T0,wT]	=	fftFourierInt( u0(xx), -pi, pi, Nmax, 'cubic', false );

b       =   b(bidx) / (2*pi);
T0      =   T0(tidx) / sqrt(2*pi);
A       =   make_A(b,D);
u       =	@(x,t) u_func_spectral(x,t,A,T0);

%{
N       =   1;
Niter	=   10;
Niter2	=   Niter/2;
Nmax	=   250;
tol     =   1e-3;
[x,t]	=   deal( linspace(-pi,pi,20).', linspace(0,Tmax,10) );
[dx,dt]	=   deal( 1e-3, 1e-3 );

b       =   integral( @(x) f(x),  -pi, pi ) / (2*pi);
T0      =   integral( @(x) u0(x), -pi, pi ) / sqrt(2*pi);

idx     =   1;
idx_best=	1;
err_list=	inf;
err_best=	inf;
local_minimum	=   false;
while (err_best > tol) && ~local_minimum && (N < Nmax)
    N       =   N + Niter;
    idx     =   idx + 1;
    
    [b,T0,A]=	iter_spectral(D,f,u0,b,T0,N);
    u       =	@(x,t) u_func_spectral(x,t,A,T0);
    
%     U	=   u(x,t);
%     err	=	bsxfun( @times, -f(x), U ); % -f*u
%     err	=   err + D * (u(x+dx,t)-2*U+u(x-dx,t))/(dx^2); % D*u_xx-f*u
%     err =	err - (u(x,t+dt)-u(x,t-dt))/(2*dt); % D*u_xx-f*u-u_t
%     err	=   max( abs( err(:) ) );
    
    T	=	expmv(Tmax,A,T0);
    tail=	[1:Niter2, N-Niter2+1:N]; % tails of T
    err	=   max(abs(T(tail)))/max(abs(T));
    
    err_list	=   [err_list, err];
    if err < err_best
        err_best	=	err;
        u_best      =	u;
        idx_best	=   idx;
    end
    
    if (idx_best > 2) && (length(err_list) >= idx_best+2)
        local_minimum	=   all( diff( err_list(idx_best-2:idx_best) ) < 0 ) && ...
                            all( diff( err_list(idx_best:idx_best+2) ) > 0 );
    end
end

u	=   u_best;
%}

end

function S = S_func_spectral(t,A,T0)

tsize       =   size(t);
t           =   t(:);

sqrt_2pi	=   sqrt(2*pi);
S           =   zeros(length(t),1);
for jj = 1:numel(t)
%     T       =   expm(A*t(jj)) * T0;
    T       =   expmv(t(jj),A,T0);
    S(jj)	=   sqrt_2pi * T(floor(length(T)/2)+1);
end

S           =   reshape( S, tsize );

end

function [S,b,T0,A] = S_func_spectral_iter(D,f,u0,Tmax)

% N       =   9;
% Niter	=   6;
% tol     =   1e-3;
% t       =   linspace(0,Tmax,10);

Nmax	=   128;
N0      =   floor(Nmax/2);

xx      =   linspace(-pi,pi,Nmax+1);
[b, wb]	=	fftFourierInt( f(xx),  -pi, pi, Nmax, 'trap', true  );
[T0,wT]	=	fftFourierInt( u0(xx), -pi, pi, Nmax, 'trap', false );

b       =   b / (2*pi);
T0      =   T0(N0-ceil(N0/2)+1:N0+ceil(N0/2)) / sqrt(2*pi);
A       =   make_A(b,D);
S       =	@(t) S_func_spectral(t,A,T0);

% b       =   integral( @(x) f(x),  -pi, pi ) / (2*pi);
% T0      =   integral( @(x) u0(x), -pi, pi ) / sqrt(2*pi);
% 
% [b,T0,A]=	iter_spectral(D,f,u0,b,T0,N);
% S       =	@(t) S_func_spectral(t,A,T0);
% 
% err     =	inf;
% while err > tol
%     N	=   N + Niter;
%     [S_last,b_last,T0_last]	=   deal(S,b,T0);
%     
%     [b,T0,A]=	iter_spectral(D,f,u0,b_last,T0_last,N);
%     S       =	@(t) S_func_spectral(t,A,T0);
%     
%     err	=   max( abs( S(t) - S_last(t) ) );
% end

end

%==========================================================================
% parabolicPeriodic1D method for signal
%==========================================================================

function S = S_func_pp1D(D,f,u0)



end

%==========================================================================
% Spectral method testing
%==========================================================================

function b = get_b(f,N)

M	=   N-1;
b	=   zeros(2*N-1,1);
for L = -M:M
    ii      =   L+M+1;
    b(ii)	=   integral( @(x) f(x) .* exp((1i*L)*x), -pi, pi ) / (2*pi);
end

end

function A = get_A(b,D,N)

M	=   N-1;
n	=   floor(N/2);

A	=	-1 * diag(D*(-n:n).^2);
for L = -M:M
    ii	=	L+M+1;
    A	=   A - diag( b(ii)*ones(N-abs(L),1), L );
end

end

function T0 = get_T0(u0,N)

T0	=   zeros(N,1);
n	=   floor(N/2);

for L = -n:n
    ii      =   L+n+1;
    T0(ii)	=   integral( @(x) u0(x) .* exp((-1i*L)*x), -pi, pi ) / sqrt(2*pi);
end

end

function u = u_soln(x,t,A,T0,N)

n	=   floor(N/2);
n3	=   reshape(-n:n,1,1,[]);

x	=   x(:);
t	=   t(:).';
nx	=   numel(x); 
nt	=   numel(t);
X	=   zeros(nx,1,N);
T	=   zeros(1,nt,N);

for ii = 1:nx
    X(ii,:,:)	=	(1/sqrt(2*pi)) * exp( (1i*x(ii)) * n3 );
end

for jj = 1:nt
    T(:,jj,:)	=   reshape( expm(A*t(jj)) * T0, 1, 1, [] );
end

u	=   sum( bsxfun(@times, X, T), 3 );

end

function Lu = Lu_check(x,t,u,f,D)

h	=	1e-4;
h2	=   2*h;
hsq	=   h^2;

f0      =   f(x);
u0      =   u(x,t);
u_t     =   ( u(x,t+h) - u(x,t-h) ) / h2;
u_xx	=   ( u(x-h,t) - 2*u0 + u(x+h,t) ) / hsq;

Lu      =   u_t - ( D*u_xx - f0*u0 );

end

function S = S_soln(t,A,T0)

S	=   zeros(size(t));
for ii = 1:numel(t)
    S(ii)	=   sqrt(2*pi) * sum( expm(A*t(ii)) * T0 );
end

end

function spectral_testing

N	=   81;

b	=	get_b(f,N);
T0	=	get_T0(u0,N);
A	=   get_A(b,D,N);

u	=	@(x,t)	u_soln(x,t,A,T0,N);
Lu	=   @(x,t)	Lu_check(x,t,u,f,D);
S	=	@(t)	S_soln(t,A,T0);

end