classdef Toeplitz3D
    %TOEPLITZ3D Optimized computations for 3D Block-Toeplitz matrices.
    
    properties
        
        % Matrix Properties
        m           % The size of the underlying Toeplitz matrix is [prod(m) x prod(m)]
        n           % The size of the compressed Toeplitz matrix is [prod(n) x prod(n)]
        
        % Matrix Multiplication Properties
        p           % fft pad size
        m_fft       % padded x-size
        n_fft       % padded t-size
        N_fft       % fft length
        Ft          % fft of t
        
        % Compressed matrix
        D           % (optional) Diagonal matrix to be added to Toeplitz matrix
        t           % compressed matrix
        
        % Flags
        isNearDiag	% (optional) Flag to indicate that T is nearly diagonal
        
    end
        
    methods
        
        %==================================================================
        % Constructor
        %==================================================================
        function T = Toeplitz3D( t, tsize, D, isNearDiag )
            
            % Parse compress matrix size
            if nargin < 2
                tsize	=   [];
            end
            [t,tsize]	=   parse_t_input(t,tsize);
            
            % x-size/compressed matrix size
            T.m     =   floor(tsize/2)+1;
            T.n     =   tsize;
            
            % sizes for fft of compressed matrix padded to a fast fft size
            T.p     =   nextfastfft(T.n) - T.n;
            T.n_fft	=   T.n + T.p;
            T.m_fft	=   T.m + floor(T.p/2);
            T.N_fft	=   prod(T.n_fft);
            
            % Diagonal matrix
            if nargin < 3
                D	=   [];
            end
            
            % isNearDiag flag (T is dominated by central elements of t)
            if nargin < 4 || isempty(isNearDiag)
                isNearDiag	=   false;
            end
            T.isNearDiag	=   isNearDiag;
            
            % Compressed matrix coefficient vector
            T.t     =   t(:);
            
            % Update fft of padded compressed matrix
            T       =   updateFFT(T);
            
            % Add diagonal matrix (must be done last; depends on above properties)
            T       =   addDiagonal(T,D);
            
        end
        
        %==================================================================
        % Array multiplication
        %==================================================================
        function b = times(T,x)
            
            if isa( x, 'Toeplitz3D' )
                b	=   times(x,T);
                return
            end
            
            if isscalar(x)
                b       =   T;
                b.D     =   b.D .* x;
                b.t     =   b.t .* x;
                b.Ft	=   b.Ft .* x;
            else
                error( 'Array multiplication only implemented for scalar multiplication.' );
            end
            
        end
        
        %==================================================================
        % Matrix Transpose
        %==================================================================
        function S = transpose(T)
            
            % T.t coefficients are reversed along each dimension
            S       =   T;
            tt      =   reshape( S.t, S.n );
            tt      =   tt( end:-1:1, end:-1:1, end:-1:1 );
            
            % T.D is unchanged by transpose
            S.t     =   tt(:);
            S       =   updateFFT(S);
            
        end
        
        %==================================================================
        % Matrix Conjugate Transpose
        %==================================================================
        function S = ctranspose(T)
            
            % T.t coefficients are reversed along each dimension
            S       =   T;
            tt      =   reshape( S.t, S.n );
            tt      =   conj( tt( end:-1:1, end:-1:1, end:-1:1 ) );
            
            % T.D coefficients are conjugated
            S.D     =   conj( S.D );
            S.t     =   tt(:);
            S       =   updateFFT(S);
            
        end
        
        %==================================================================
        % Matrix multiplication
        %==================================================================
        function b = mtimes(T,x)
            
            if ( isa(T,'Toeplitz3D') && isscalar(x) ) || ( isa(x,'Toeplitz3D') && isscalar(T) )
                
                %----------------------------------------------------------
                % Multiplication by scalar
                %----------------------------------------------------------
                b	=   T .* x;
                
            elseif isa(T,'Toeplitz3D') && check_input_size(x,T)
                
                %----------------------------------------------------------
                % RHS Array Multiplication
                %----------------------------------------------------------
                % Reshape x to 3D
                xsize	=   size(x);
                if ~isequal( xsize, T.m )
                    x	=   reshape(x,T.m);
                end
                
                % Pad x with zeros
                b                               =	zeros(T.n_fft);
                b(1:T.m(1),1:T.m(2),1:T.m(3))	=   x;
                b                               =	b(:);
                
                % Convolve toeplitz matrix T with x
                b	=	fft(b,T.N_fft);
                b	=	T.Ft .* b;
                b	=	ifft(b,T.N_fft);
                
                % Reshape b
                b	=   reshape(b,T.n_fft);
                b	=   b(1:T.n_fft(1),1:T.n_fft(2),1:T.n_fft(3));
                b	=   ifftshift(b);
                
                % Extract result with unpadded size T.m
                b	=   b(  floor(T.p(1)/2) + (1:T.m(1)), ...
                            floor(T.p(2)/2) + (1:T.m(2)), ...
                            floor(T.p(3)/2) + (1:T.m(3))	);
                
                % Add contributions of D matrix if necessary, reshape, and return
                if ~isequal( size(b), xsize )
                    if ~isempty( T.D )
                        b	=   b + reshape( T.D, T.m ) .* x;
                    end
                    
                    b	=   reshape( b, xsize );
                else
                    if ~isempty( T.D )
                        b	=   b + reshape( T.D, T.m ) .* x;
                    end
                    
                    b	=   reshape( b, xsize );
                end
                
            elseif isa(x,'Toeplitz3D') && check_input_size(T,x)
                
                %----------------------------------------------------------
                % LHS Array Multiplication
                %----------------------------------------------------------
                % Use fact that x*A = (A'*x')'
                b	=   ( x' * T' )';
                
            else
                error( 'Unsupported multiplication (*) type.' );
            end
            
        end
        
        %==================================================================
        % Matrix Trace
        %==================================================================
        function Tr = trace(T)
            
            mid     =   floor(T.n/2) + 1;
            mid_idx	=   sub2ind( T.n, mid(1), mid(2), mid(3) );
            
            Tr      =	T.t(mid_idx) * size(T,1);
            if ~isempty( T.D )
                Tr	=   Tr + sum( T.D(:) );
            end
            
        end
        
        %==================================================================
        % Shift diagonal term by constant
        %==================================================================
        function T = shift(T,mu)
            
            mid     =   floor(T.n/2) + 1;
            mid_idx	=   sub2ind( T.n, mid(1), mid(2), mid(3) );
            
            T.t(mid_idx)	=	T.t(mid_idx) - mu;
            T               =	updateFFT(T);
            
            % Note: don't need to shift D! (T+D)-mu*I = (T-mu*I) + D
            
        end
        
        %==================================================================
        % Add general diagonal matrix
        %==================================================================
        function T = updateFFT(T)
            T.Ft                                =	zeros(T.n_fft);
            T.Ft(1:T.n(1),1:T.n(2),1:T.n(3))	=	reshape(T.t,T.n);
            T.Ft                                =   flipud(T.Ft(:));
            T.Ft                                =	fft(T.Ft,T.N_fft);
        end
        
        %==================================================================
        % Add general diagonal matrix
        %==================================================================
        function T = addDiagonal(T,D)
            
            D       =   parse_D_input(D,T.m);
            if isscalar(D)
                % Constant diagonal matrix; just shift DC term of T.t
                T	=   shift(T,-D);
            else
                D	=   cast( D, 'like', T.t );
                if isempty(T.D)
                    T.D     =   D;
                else
                    T.D     =   T.D + D;
                end
            end
            
        end
        
        %==================================================================
        % Matrix 1-norm
        %==================================================================
        function Tnorm = norm(T,p)
            
            if nargin < 2; p = 1; end
            if p ~= 1, error( 'Toeplitz Matrix Norm only implemented for p = 1.' ); end
            
            get_ns	=   @(n) -floor(n/2):ceil(n/2)-1;
            nx      =   get_ns(T.m(1));
            ny      =   get_ns(T.m(2));
            nz      =   get_ns(T.m(3));
            mid     =   floor(T.n/2)+1;
            midx	=   sub2ind(T.n,mid(1),mid(2),mid(3));
            
            if prod(T.n) <= 32^3
                
                %----------------------------------------------------------
                % Matrix is sufficiently small; compute exact 1-norm
                %----------------------------------------------------------
                if ~isempty(T.D)
                    tt      =	T.t;
                end
                
                Tnorm	=   -inf;
                col     =   0;
                for kk = 1-nz(1):T.n(3)-nz(end)
                    for jj = 1-ny(1):T.n(2)-ny(end)
                        for ii = 1-nx(1):T.n(1)-nx(end)
                            
                            col	=   col + 1;
                            idx	=   plus3D( ii+nx, T.n(1)*(jj+ny-1), T.n(1)*T.n(2)*(kk+nz-1) );
                            
                            if isempty(T.D)
                                Tn          =	sum(abs(T.t(idx(:))));
                            else
                                tt(midx)	=   T.t(midx) + T.D(col);
                                Tn          =	sum(abs(tt(idx(:))));
                            end
                            
                            if Tn > Tnorm
                                Tnorm	=	Tn;
                            end
                            
                        end
                    end
                end
                
            else
                
                %----------------------------------------------------------
                % Matrix is too large; compute approximate 1-norm
                %----------------------------------------------------------
                if T.isNearDiag
                    % T.t is dominated by central elements; approximate 1-norm by:
                    %	Tnorm	=	sum(abs([central elements])) + abs(T.D([middle]))
                    [ii,jj,kk]	=	ndgrid( mid(1)+nx, mid(2)+ny, mid(3)+nz );
                    idx         =	sub2ind(T.n,ii(:),jj(:),kk(:));
                    Tnorm       =	sum(abs(T.t(idx)));
                    if ~isempty(T.D)
                        Dmid        =   floor(T.m/2)+1;
                        Dmidx       =   sub2ind(T.m,Dmid(1),Dmid(2),Dmid(3));
                        Tnorm       =   Tnorm + abs(T.D(Dmidx));
                    end
                else
                    % Approximate 1-norm generically
                    mpow	=   1;
                    Tnorm	=   normAm(T,mpow);
                end
            end
            
        end
        
        %==================================================================
        % Matrix Size
        %==================================================================
        function Tsize = size(T,dim)
            if nargin < 2
                Tsize	=   length(T)*[1,1];
            elseif dim == 1 || dim == 2
                Tsize	=   length(T);
            elseif dim > 2 && dim == round(dim)
                Tsize	=   1;
            else
                error('Invalid dimension.');
            end 
        end
        
        %==================================================================
        % Matrix Absolute Value
        %==================================================================
        function S = abs(T)
            if isempty( T.D )
                S	=	Toeplitz3D( abs(T.t), T.n );
            else
                tt          =   T.t;
                DD          =   T.D;
                mid         =   floor(T.n/2)+1;
                midx        =   sub2ind(T.n,mid(1),mid(2),mid(3));
                
                DD          =   abs( DD + tt(midx) );
                tt(midx)    =	0.0;
                tt          =   abs(tt);
                
                S	=	Toeplitz3D( tt, T.n, DD );
            end
        end
                
        %==================================================================
        % Matrix Length
        %==================================================================
        function Tlen = length(T)
            Tlen	=	prod(T.m);
        end
        
        %==================================================================
        % Matrix Class
        %==================================================================
        function Tclass = class(T)
            Tclass	=	class(T.t);
        end
        
        %==================================================================
        % isfloat
        %==================================================================
        function bool = isfloat(T)
            bool	=   true;
        end
        
        %==================================================================
        % ismatrix
        %==================================================================
        function bool = ismatrix(T)
            bool	=   true;
        end
        
        %==================================================================
        % isreal
        %==================================================================
        function bool = isreal(T)
            bool	=   isreal(T.t);
        end
        
        %==================================================================
        % isequal
        %==================================================================
        function bool = isequal(T,S)
            % Quick checks
            bool	=	isequal( T.m, S.m )         &&	...
                        isequal( T.n, S.n )         &&	...
                        isequal( T.p, S.p )         &&	...
                        isequal( T.m_fft, S.m_fft )	&&  ...
                        isequal( T.n_fft, S.n_fft )	&&  ...
                        isequal( T.N_fft, S.N_fft )	&&  ...
                        isequal( isempty(T.D), isempty(S.D) );
            
            % Check actual values
            if bool
                if isempty( T.D )
                    bool	=	bool && isequal( T.t, S.t ) && isequal( T.Ft, S.Ft );
                else
                    mid     =   floor(T.n/2)+1;
                    midx	=   sub2ind(T.n,mid(1),mid(2),mid(3));
                    
                    % Note that we do not check if T.Ft, S.Ft are equal, as
                    % it may be the case that T.D and S.D have absorbed their
                    % respective DC terms T.t(midx), S.t(midx); only the
                    % sums T.t(midx)+T.D and S.t(midx)+S.D are compared
                    bool	=	bool && ...
                        isequal( T.t([1:midx-1,midx+1:end]), S.t([1:midx-1,midx+1:end]) )	&&	...
                        isequal( T.t(midx) + T.D, S.t(midx) + S.D );
                end
            end
        end
        
    end
    
end

function bool = check_input_size(x,tsize)

if isa(tsize,'Toeplitz3D')
    tsize	=   tsize.m;
end

Nt      =   prod( tsize );
bool	=   (	isequal( size(x), tsize )	||	...
                isequal( size(x), [Nt,1] )  ||  ...
                isequal( size(x), [1,Nt] )	);

end

function [t,tsize] = parse_t_input(t,tsize)

if isempty(tsize)
    if ~isvector(t)
        tsize	=   size(t);
    else
        len     =   length(t);
        width	=   round(len^(1/3));
        if len ~= width^3
            error( 'If tsize is unspecified, t must be a vector of length M^3.' );
        else
            tsize	=	[width,width,width];
        end
    end
elseif numel(t) ~= prod(tsize)
    error(  'Number of elements given by tsize (%d) does not match t (%d)', ...
            prod(tsize), numel(t) );
end

if ~isequal( size(t), tsize )
    t	=   reshape(t,tsize);
end

end

function D = parse_D_input(D,m)

if isempty(D) || ~any(D(:))
    D	=   [];
    return
end

if ~( isfloat(D) && ismatrix(D) )
    error( 'Diagonal matrix D must be a floating point array.' );
end

M       =	prod(m);
[dx,dy]	=   size(D);

if dx == 1 && dy == 1
    % D is scalar
    return
end

if dx == 1 || dy == 1
    % D is a row/dolumn vector
    D	=   D(:);
    dx	=   length(D);
    dy	=   1;
end

if ~( ( dx == M && dy == 1 ) || ( dx == M && dy == M ) )
    error( 'Diagonal matrix must be a column vector or have the same size as T.' );
end

if dx == dy
    if ~isdiag( D )
        warning( 'D is not diagonal; taking only the diagonal part.' );
    end
    D	=   diag(D);
end

if issparse( D )
    D	=   full(D);
end

end
