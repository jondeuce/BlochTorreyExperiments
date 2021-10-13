function [ S, u, b, T0, A ] = heatLikeEqn3D( ...
    D1, D2, D3, f, u0, wmax, xb, yb, zb, Mx, My, Mz, order, tflip, loc, Tmax )
%HEATLIKEEQN3D Solves the 1D generalized heat-like equation on the domain
% [-pi,pi]^3 x [0,Tmax]. The equation being solved is:
%       
%   u_t = D1*u_xx + D2*u_yy + D3*u_zz - f(x)*u
% 
% If the argument 'tflip' is passed, the solution will be 'flipped' at time
% t = tflip by taking the complex conjugate of u(x,y,z,tflip) and then
% further propogated from this state.

%==========================================================================
% Input parsing
%==========================================================================

if nargin < 16 || isempty(Tmax)
    Tmax	=   1.0;
end

isInt	=   false;
if nargin < 15 || isempty(loc)
    loc         =   'endpoints';
end
if strcmpi( loc, 'interior' )
    fxb         =   @(xb,Mx) xb + diff(xb)/(Mx+1)*[0.5,-0.5];
    [xb,yb,zb]	=   deal( fxb(xb,Mx), fxb(yb,My), fxb(zb,Mz) );
    isInt       =	true;
end

% support function input
isFunc_f	=   isa( f,  'function_handle' );
isFunc_u0	=   isa( u0, 'function_handle' );

if isFunc_f || isFunc_u0
    [x,y,z]	=   deal(   linspace(xb(1),xb(2),Mx+1).',    ...
                        linspace(yb(1),yb(2),My+1).',    ...
                        linspace(zb(1),zb(2),Mz+1).'	);
    if isFunc_f
        f	=   f(x,y,z);
    end
    if isFunc_u0
        u0	=   u0(x,y,z);
    end
else
    if isempty(Mx) || isempty(My) || isempty(Mz)
        [Mx,My,Mz]	=   dealArray( size(u0)-1 );
    end
end

% wmax determines the max frequency component in the solution
%   Note:	Number of elements in square A matrix will be (2*wmax+1)^6!
%           For numel(A) = numel(u0): wmax = floor(((Mx*My*Mz)^(1/6)-1)/2)
if nargin < 6
    wmax	=   floor(((Mx*My*Mz)^(1/6)-1)/2);
else
    wmax	=   round(wmax);
end

% Order of integration ('trapezoidal' or 'cubic')
if nargin < 13
    order	=   'cubic';
end

% Flip time (default none)
if nargin < 14
    tflip	=   [];
elseif ~isempty( tflip ) && ( ~isreal(tflip) || ( tflip < 0 ) )
    warning( 'Parameter tflip must be real and non-negative' );
    tflip	=   [];
elseif tflip == 0
    u0      =   conj(u0);
    tflip	=   [];
end

%==========================================================================
% Solve the PDE via spectral expansion:
%   u(x,y,z,t) = sum( T(t) * exp( i*(wx*x+wy*y+wz*z) ), wx, wy, wz )
%==========================================================================

% Spectral method
[u,S,b,T0,A]	=	spectralSolve(D1,D2,D3,f,u0,wmax,xb,yb,zb,Mx,My,Mz,order,tflip,loc,isInt);

end

%==========================================================================
% Spectral method for signal
%==========================================================================

function [u,S,b,T0,A] = spectralSolve(D1,D2,D3,f,u0,wmax,xb,yb,zb,Mx,My,Mz,order,tflip,loc,isInt)
% Problem is solved on the domain [-pi,pi]^3 and transformed back

% if wmax <= 7,	buildSparse = false;
% else            buildSparse = true;
% end
buildSparse	=   false;

% Calculate frequency components of f
[b,nx,ny,nz]	=	...
    fftFourierInt3D(f, [-pi,pi],[-pi,pi],[-pi,pi],Mx,My,Mz,order,true,loc);

% Calculate frequency components of u0 (frequencies nx,ny,nz are identical)
[T0,~,~,~]	=	...
    fftFourierInt3D(u0,[-pi,pi],[-pi,pi],[-pi,pi],Mx,My,Mz,order,false,loc);

% Truncate b and T0 to keep only necessary terms, and scale by constants
[b,T0,m]	=   handle_b_T0(b,T0,nx,ny,nz,wmax,buildSparse);

% Build A matrix
A	=	build_A(b,D1,D2,D3,nx,ny,nz,wmax,xb,yb,zb,m,buildSparse);

% Check for flip time
T1	=	handle_tflip(A,T0,tflip,buildSparse);

% Dimensions of containing box (less than xb,yb,zb if points are interior)
[dx,dy,dz]	=   get_dxdydz(xb,yb,zb,Mx,My,Mz,isInt);

% Get final function handles
u	=   @(x,y,z,t)	u_func(x,y,z,t,xb,yb,zb,A,T0,T1,tflip);
S	=   @(t)        S_func(t,dx,dy,dz,A,T0,T1,tflip);

end

function [b,T0,m] = handle_b_T0(b,T0,nx,ny,nz,wmax,buildSparse)

%==========================================================================
% Calculate frequencies to keep
%==========================================================================
mid     =   [find(nx==0),find(ny==0),find(nz==0)]; % subcripts of DC term
% wrange	=   @(n) -floor(n/2):ceil(n/2)-1;
% getw	=	@(wmax,siz,ii) wrange(min(2*wmax+1,siz(ii)));
% wb      =   getw(2*wmax,size(b),1);
% bsize	=   [length(wb),length(wb),length(wb)];
% tsize	=   floor(bsize/2)+1;
% wt      =	getw(wmax,tsize,1);
if 4*wmax+1 > min(size(b)); wmax = floor((min(size(b))-1)/4); end
wb	=   -2*wmax:2*wmax;
wt	=   -wmax:wmax;

%==========================================================================
% Truncate T0 to appropriate size
%==========================================================================

% Only need [-wmax,...,wmax] freqs for T0 and to scale down by (2*pi)^(3/2)
scale	=   1/(2*pi)^1.5;
T0      =   scale * T0( mid(1)+wt, mid(2)+wt, mid(3)+wt );
if buildSparse, T0 = double(T0); end

%==========================================================================
% Truncate b, keeping principle frequency components
%==========================================================================

if ~buildSparse
    
    %======================================================================
    % Non-sparse version
    %======================================================================
    
    % Need frequencies in range [-(2*wmax), ..., 2*wmax] for b, and to
    % scale down by (2*pi)^3
    scale	=   1/(2*pi)^3.0;
    b       =   scale * b( mid(1)+wb, mid(2)+wb, mid(3)+wb );
    m       =   [];
    
else
    
    %======================================================================
    % Sparse version
    %======================================================================
    
    % Truncate b to wmax
    scale	=   1/(2*pi)^3.0;
    b       =   scale * b( mid(1)+wb, mid(2)+wb, mid(3)+wb );
    
    % Keep largest frequency components
    bmid        =   floor(size(b)/2)+1; % subcripts of DC term
    [~,bidx]	=	sort(abs(b(:)));
    len         =   300;
    bidx        =   bidx(end-len+1:end);
    m           =   false(size(b));
    m(bidx)     =   true;
    
    % Keep center frequencies
    wmids       =   -min(3,wmax):min(3,wmax);
    m(bmid(1)+wmids,bmid(2)+wmids,bmid(3)+wmids)	=   true;
    
    % Mask out small values of b
    b(~m)       =   0;
    
end

end

function A = build_A(b,D1,D2,D3,nx,ny,nz,wmax,xb,yb,zb,m,buildSparse)

tsize	=   floor(size(b)/2) + 1;
wrange	=   @(n) -floor(n/2):ceil(n/2)-1;
getw	=	@(wmax,siz,ii) wrange(min(2*wmax+1,siz(ii)));
ws      =	getw(wmax,tsize,1);
M       =   length(ws);
M2      =   ceil(M/2);

[m_x,m_y,m_z]	=   deal( find(nx==0)+ws, find(ny==0)+ws, find(nz==0)+ws );
[nx,ny,nz]	=   deal( nx(m_x), ny(m_y), nz(m_z) );
bmid        =	floor(size(b)/2)+1;

if ~buildSparse
    
    %======================================================================
    % Build Full A-matrix
    %======================================================================
    
    if ( true )
        % Compressed Toeplitz-Form of A-matrix
        isNearDiag	=   true;
        A           =   Toeplitz3D(-b,[],[],isNearDiag);
        
    else
        % Build Full A-matrix
        A           =   zeros(M^3,M^3,'like',b);
        for wx = ws
            for wy = ws
                for wz = ws
                    [ii,jj,kk]	=   dealArray( [wx,wy,wz]+M2 );
                    idx         =   ii + M*( jj-1 + M*(kk-1) );
                    A(idx,:)	=   A(idx,:) - ...
                        reshape( b( bmid(1)+ws-wx, bmid(2)+ws-wy, bmid(3)+ws-wz ), 1, M^3 );
                end
            end
        end
    end
    
else
    
    %======================================================================
    % Build Sparse A-matrix
    %======================================================================
    
    nnz_b	=   nnz(abs(b));
    chunk	=   zeros(ceil(nnz_b/2)*M^3,1);
    iinds	=   chunk;
    jinds	=   chunk;
    bb      =   chunk;
    Asize	=   floor(size(b)/2)+1;
    inds	=   (1:prod(Asize)).';
    binds	=   reshape(1:numel(b),size(b));
    count	=   0;
    
    for wx = ws
        for wy = ws
            for wz = ws
                [ii,jj,kk]	=   dealArray( [wx,wy,wz]+M2 );
                idx         =   ii + M*( jj-1 + M*(kk-1) );
                
                [m_x,m_y,m_z]	=   deal( bmid(1)+ws-wx, bmid(2)+ws-wy, bmid(3)+ws-wz );
                mm          =   m(m_x,m_y,m_z);
                jj          =	inds(mm);
                
                if ~isempty(jj)
                    jjb     =   binds(m_x,m_y,m_z);
                    jjb     =   jjb(mm(:));
                    njj     =   numel(jj);
                    count	=   count + njj;
                    
                    if count > numel(bb)
                        iinds	=   [ iinds; chunk ];
                        jinds	=   [ jinds; chunk ];
                        bb      =   [ bb; chunk ];
                    end
                    
                    iinds(count-njj+1:count)	=   idx;
                    jinds(count-njj+1:count)	=   jj;
                    bb(count-njj+1:count)       =	b(jjb);
                end
            end
        end
    end
    
    iinds	=   iinds(1:count);
    jinds	=   jinds(1:count);
    bb      =   bb(1:count);
    
    A       =	sparse(iinds,jinds,double(-bb),M^3,M^3,numel(iinds));
    
end

%==========================================================================
% Add diffusion elements
%==========================================================================
scale       =   @(xb) (2*pi/diff(xb))^2;
[d1,d2,d3]	=   deal( scale(xb)*D1, scale(yb)*D2, scale(zb)*D3 );
lambda      =	plus3D( d1*(nx.*nx), d2*(ny.*ny), d3*(nz.*nz) );
if ~buildSparse
    if isa( A, 'Toeplitz3D' )
        % For Toeplitz A matrix
        A	=	addDiagonal(A,-lambda(:));
    else
        % For full A matrix
        A	=	A + diag(-lambda(:),0);
    end
else
    A	=	A + spdiags(-lambda(:),0,M^3,M^3);
end
clear lambda

%==========================================================================
% Testing (brute force build)
%==========================================================================
%{
for m_x = 1:M
    for m_y = 1:M
        for m_z = 1:M
            for n_x = 1:M
                for n_y = 1:M
                    for n_z = 1:M
                        ii	=   m_x+M*(m_y-1+M*(m_z-1));
                        jj	=   n_x+M*(n_y-1+M*(n_z-1));
                        
                        A(ii,jj)	=   A(ii,jj) - ...
                            b( b0(1)+n_x-m_x, b0(2)+n_y-m_y, b0(3)+n_z-m_z );
                    end
                end
            end
        end
    end
end

getA	=	@(A,mx,my,mz,nx,ny,nz) ...
    A(  (mx+M2)+M*((my+M2)-1+M*((mz+M2)-1)), ...
        (nx+M2)+M*((ny+M2)-1+M*((nz+M2)-1))     );

getb	=   @(b,lx,ly,lz) ...
    b(	b0(1)+lx, b0(2)+ly, b0(3)+lz      );
    
checkA	=   @(b,D1,D2,D3,mx,my,mz,nx,ny,nz) ...
    (-D1*mx^2-D2*my^2-D3*mz^2)*(mx==nx && my==ny && mz==nz) - ...
    getb( b, nx-mx, ny-my, nz-mz );
%}

end

function T1 = handle_tflip(A,T0,tflip,buildSparse)

if isempty(tflip)
    T1	=   [];
    return
end

T1	=   conj( expmv(tflip,A,T0(:),[],'double',true,true) );
T1	=   T1(end:-1:1,end:-1:1,end:-1:1);
if buildSparse, T1 = double(T1); end

end

function [dx,dy,dz] = get_dxdydz(xb,yb,zb,Mx,My,Mz,isInt)
% Returns the dimensions of the containing box

[dx,dy,dz]	=   deal( diff(xb), diff(yb), diff(zb) );

if isInt
    % Distance between interior endpoints is one unit less than box widths
    [dx,dy,dz]	=   deal( dx+dx/Mx, dy+dy/My, dz+dy/Mz );
end

end

function u = u_func(x,y,z,t,xb,yb,zb,A,T0,T1,tflip)

t           =   t(:);
nmax        =   size(T0)-ceil(size(T0)/2);
[nx,ny,nz]	=   deal( -nmax(1):nmax(1), -nmax(2):nmax(2), -nmax(3):nmax(3) );
scale       =   1/(2*pi)^1.5;
isFlip      =	~( isempty(tflip) || isempty(T1) );

% Scale and shift to local frame
SAS         =	@(x,xb) (2*pi/diff(xb))*x - (pi*sum(xb)/diff(xb));

for tidx = 1:numel(t)
    
    if isFlip && ( t(tidx) >= tflip )
        T	=	expmv( t(tidx)-tflip, A, T1(:), [], 'double', true, true );
    else
        T	=	expmv( t(tidx), A, T0(:), [], 'double', true, true );
    end
    T       =   reshape( T, size(T0) );
    
    for ii = 1:numel(nx)
        for jj = 1:numel(ny)
            for kk = 1:numel(nz)
                
                sT	=	scale .* T(ii,jj,kk);
                
                if all( [tidx,ii,jj,kk] == 1 )
                    
                    % First pass, initialize u
                    u	=   sT .* exp( complex( 0, plus3D( nx(ii)*SAS(x,xb), ny(jj)*SAS(y,yb), nz(kk)*SAS(z,zb) ) ) );
                    
                    if numel(t) > 1
                        u	=   cat( 4, u, zeros( [size(u),numel(t)-1], 'like', u ) );
                    end
                    
                else
                    
                    u(:,:,:,tidx)	=   u(:,:,:,tidx) + ...
                        sT .* exp( complex( 0, plus3D( nx(ii)*SAS(x,xb), ny(jj)*SAS(y,yb), nz(kk)*SAS(z,zb) ) ) );
                    
                end
                                
            end
        end
    end
    
end

end

function S = S_func(t,dx,dy,dz,A,T0,T1,tflip)

tol         =   2^(-24); % single precision floating error
tsize       =   size(t);
t           =   t(:);
V          	=	(dx*dy*dz); % relative volumes
scale       =   V * (2*pi)^1.5;
isFlip      =   ~( isempty(tflip) || isempty(T1) );

S           =   zeros(numel(t),1,'double');
DC_sub      =   ceil(size(T0)/2);
DC_ind      =   sub2ind( size(T0), DC_sub(1), DC_sub(2), DC_sub(3) );

for tidx = 1:numel(t)
    
    if isFlip && ( t(tidx) >= tflip )
        T	=	expmv( t(tidx)-tflip, A, T1(:), [], 'double', true, false, false, false, [], tol );
    else
        T	=	expmv( t(tidx), A, T0(:), [], 'double', true, false, false, false, [], tol );
    end
    
    S(tidx)	=   scale * T(DC_ind);
    
end

S           =   reshape( S, tsize );

end
