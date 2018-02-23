classdef BlochTorreyOp
    %BLOCHTORREYOP Class representing the discrete Bloch-Torrey equation
    % operator w/ 2nd order central difference approximations on
    % derivatives and with periodic boundary conditions. The operator is:
    %
    %   L[x]  =  div(D*grad(x)) - Gamma*x
    %         =  D*lap(x) - Gamma*x          % for scalar D
    %
    % The corresponding parabolic PDE (the Bloch-Torrey equation) is then:
    %   dx/dt =  L[x]
    %         =  D*lap(x) - Gamma*x
    %
    % Currently, D must be a scalar and Gamma must be a 3D array of the
    % same size as x.
    
    properties ( GetAccess = public, SetAccess = immutable )
        gsize % Grid size, e.g. [512,512,512]
        gdims % Grid dimensions (unitful), e.g. [3000,3000,3000] um
        N     % Total number of elements on grid, i.e. prod(gsize)
        h     % Physical distance between elements, e.g. gdims./gsize = [5.8594,5.8594,5.8594] um
    end
    
    properties ( GetAccess = public, SetAccess = private )
        D      % Diffusion coefficient, e.g. 3037 um^2/s
    end
    
    properties ( Dependent = true, GetAccess = public, SetAccess = private )
        Diag   % Diagonal of operator = -6*D/h^2 - Gamma
        Gamma  % Complex decay coefficient = -6*D/h^2 - Diag
        R2map  % Local R2 value = real(Gamma)
        dOmega % Dephasing frequencies = imag(Gamma)
    end
    
    properties ( GetAccess = private, SetAccess = private )
        buffer % Data buffer for storing either Diag or Gamma
        state  % State which determines what data buffer represents
    end
    
    properties ( Constant )
        DiagState  = 1 % Data stored in buffer is Diag
        GammaState = 2 % Data stored in buffer is Gamma
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CLASS CONSTRUCTOR:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        function [ A ] = BlochTorreyOp( Gamma, Dcoeff, GridSize, GridDims )
            % [ A ] = BlochTorreyOp( Gamma, Dcoeff, GridSize, GridDims )
            %   INPUTS:
            %       Gamma:      Complex decay array
            %       Dcoeff:     Diffusion coefficient [um^2/s == 1000 * mm^2/ms], e.g. water ~ 3037 um^2/s
            %       GridSize:   Grid size, e.g. [512,512,512]
            %       GridDims:   Grid dimensions, e.g. [3000,3000,3000] (in um)
            
            if ~(isscalar(Gamma) || isequal(numel(Gamma), prod(GridSize)))
                error('Gamma must be scalar or have the same number of elements as the grid');
            end
            
            if ~isscalar(Dcoeff)
                error('Diffusion coefficient must be a scalar constant');
            end
            
            if ~isequal(size(GridSize),[1,3]) || ~isequal(size(GridDims),[1,3])
                error('gsize and gdims must have size [1,3], corresponding to a 3D grid');
            end
            
            if maxabs(diff(GridDims./GridSize)) > 5*eps(max(GridDims))
                error('Currently, grid size must be isotropic');
            end
            
            if any([GridDims,GridSize]) <= 0 || ~isequal(GridSize,round(GridSize))
                error('gdims and gsize must contain only positive values, and gsize must be integers');
            end
            
            A.gsize  = GridSize;
            A.gdims  = GridDims;
            A.N      = prod(A.gsize);
            A.D      = Dcoeff;
            A.h      = A.gdims./A.gsize;
            A.buffer = -2*A.D*sum(1./A.h.^2) - Gamma;
            A.state  = BlochTorreyOp.DiagState;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GETTERS/SETTERS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        function diag = get.Diag(A)
            if A.state == BlochTorreyOp.DiagState
                diag = A.buffer;
            else % BlochTorreyOp.GammaState
                diag = -2*A.D*sum(1./A.h.^2) - A.buffer;
            end
        end
        function gamma = get.Gamma(A)
            if A.state == BlochTorreyOp.DiagState
                gamma = -2*A.D*sum(1./A.h.^2) - A.buffer;
            else % BlochTorreyOp.GammaState
                gamma = A.buffer;
            end
        end
        
        function A = set.Diag(A,diag)
            A.buffer = diag;
            A.state = BlochTorreyOp.DiagState;
        end
        function A = set.Gamma(A,gamma)
            A.buffer = gamma;
            A.state = BlochTorreyOp.GammaState;
        end
        
        function A = switchbuffer(A,State)
            if isequal(State, A.state)
                return % already in state State, do nothing
            end
            switch State
                case BlochTorreyOp.DiagState % in GammaState; switch to DiagState
                    A.buffer = -2*A.D*sum(1./A.h.^2) - A.buffer;
                    A.state = BlochTorreyOp.DiagState;
                case BlochTorreyOp.GammaState % in DiagState; switch to GammaState
                    A.buffer = -2*A.D*sum(1./A.h.^2) - A.buffer;
                    A.state = BlochTorreyOp.GammaState;
            end
        end
        
        function r2 = get.R2map(A)
            r2 = real(A.Gamma);
        end
        function dw = get.dOmega(A)
            dw = imag(A.Gamma);
        end
        function A = set.R2map(A,r2)
            A.Gamma = complex(r2, A.dw);
        end
        function A = set.dOmega(A,dw)
            A.Gamma = complex(A.r2, dw);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PUBLIC METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % OVERLOADED METHODS:
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ y ] = mtimes( A, x )
            AIsBTOp = isa(A, 'BlochTorreyOp');
            xIsBTOp = isa(x, 'BlochTorreyOp');
            
            if AIsBTOp && xIsBTOp
                error('Composition of BlochTorreyOp''s is not implemented');
            elseif  AIsBTOp && ~xIsBTOp
                if isscalar(x) && isnumeric(x)
                    y = times( A, x ); %A should act like a MATRIX, not an OPERATOR
                else
                    if A.D == 0
                        if A.Diag == 1
                            y = x;
                        else
                            if size(A,2) == numel(x)
                                y = reshape(A.Diag, size(x)) .* x;
                            else
                                ncols = round(numel(x)/size(A,2));
                                if ncols * size(A,2) ~= numel(x)
                                    error('x has incorrect size.'); %TODO better error msg
                                end
                                if ndims(x) == 4
                                    sizx = size(x);
                                    y = bsxfun(reshape(A.Diag,sizx(1:3)),x);
                                else
                                    y = bsxfun(@times,A.Diag(:),x);
                                end
                            end
                        end
                    else
                        y = BlochTorreyAction(x, A.h, A.D, A.Diag, A.gsize);
                    end
                end
            elseif ~AIsBTOp &&  xIsBTOp
                if isscalar(A) && isnumeric(A)
                    y = times( A, x );
                else
                    % A*x = (x' * A')', x is the BT operator
                    isrealx = isreal(x);
                    isrealA = isreal(A);
                    if isrealx && isrealA
                        %(x' * A')' = (x * A.').'
                        y = reshape( x * A(:), size(A) );
                    elseif  isrealx && ~isrealA
                        %(x' * A')' = (x * A')'
                        y = reshape( conj(x * conj(A(:))), size(A) );
                    elseif ~isrealx &&  isrealA
                        %(x' * A')' = (x' * A.')'
                        y = reshape( conj(x' * A(:)), size(A) );
                    else % ~isrealx && ~isrealA
                        %(x' * A')'
                        y = reshape( conj(x' * conj(A(:))), size(A) );
                    end
                end
            end
        end
        
        function [ y ] = times( A, x )
            AIsBTOp = isa(A, 'BlochTorreyOp');
            xIsBTOp = isa(x, 'BlochTorreyOp');
            
            if AIsBTOp && xIsBTOp
                if iscompatible( A, x ) && isisotropic( A ) %isotropic required, else can't have single equivalent D
                    y = A;
                    if ~isequal( size(y.Diag), size(x.Diag) )
                        y.Diag = reshape(y.Diag, size(x.Diag));
                    end
                    y.Diag = y.Diag .* x.Diag; %simply multiply diagonals
                    y.D    = (y.D / y.h(1))^2; %off-diags are each D/h^2; (D/h^2)^2 = (D^2/h^2)/h^2 = D_new/h^2
                else
                    error('Cannot BlochTorreyOp''s must have the same grid size, physical dimensions, and be isotropic');
                end
            elseif AIsBTOp && ~xIsBTOp
                if isscalar(x) && isnumeric(x)
                    y      = A;
                    y.D    = y.D .* x;
                    y.Diag = y.Diag .* x;
                else
                    error('Only allowed scalar multiplication on RHS of BlochTorreyOp');
                end
            elseif ~AIsBTOp && xIsBTOp
                if isscalar(A) && isnumeric(A)
                    y      = x;
                    y.D    = A .* y.D;
                    y.Diag = A .* y.Diag;
                else
                    error('Only allowed scalar multiplication on LHS of BlochTorreyOp');
                end
            end
        end
        
        function [ y ] = plus( A, B )
            AIsBTOp = isa(A, 'BlochTorreyOp');
            BIsBTOp = isa(B, 'BlochTorreyOp');
            
            if AIsBTOp && BIsBTOp
                if ~(isequal(size(A),size(B)) && isequal(A.gsize,B.gsize) && isequal(A.gdims,B.gdims))
                    error('PLUS: Dimension mismatch');
                end
                y = A;
                y.Diag = y.Diag + B.Diag;
                y.D    = y.D    + B.D;
            elseif  AIsBTOp && ~BIsBTOp
                if isequal(size(A),size(B)) && isdiag(B)
                    y = A;
                    y.Diag = y.Diag + reshape(full(diag(B)),size(y.Diag));
                else
                    error('PLUS: second argument must be a BlochTorreyOp or a diagonal matrix.');
                end
            elseif ~AIsBTOp &&  BIsBTOp
                if isequal(size(A),size(B)) && isdiag(A)
                    y = B;
                    y.Diag = reshape(full(diag(A)),size(y.Diag)) + y.Diag;
                else
                    error('PLUS: first argument must be a BlochTorreyOp or a diagonal matrix.');
                end
            end
        end
        
        function [ y ] = minus( A, B )
            y = plus( A, -B );
        end
        
        function [ B ] = uplus( A )
            %do nothing
            B = A;
        end
        
        function [ B ] = uminus( A )
            B = A;
            B.D = -B.D;
            B.Diag = -B.Diag;
        end
        
        function [ B ] = transpose( A )
            % it's symmetric; return input
            B = A;
        end
        
        function [ B ] = ctranspose( A )
            B = A;
            if ~isreal(B)
                B.Diag = conj(B.Diag);
            end
        end
        
        function [ d ] = diag( A, k )
            if nargin == 2 && k ~= 0
                error('DIAG(::BlochTorreyOp,k~=0): Can only return 0th (i.e. main) diagonal');
            end
            
            d = A.Diag(:);
            if isscalar(d)
                d = d*ones(A.N,1);
            end
        end
        
        function [ Tr ] = trace( A )
            if isscalar(A.Diag)
                Tr = A.Diag * A.N;
            else
                Tr = sum(A.Diag(:));
            end
        end
        
        function [ B ] = abs( A )
            B = A;
            B.Diag = abs(B.Diag);
            B.D = abs(B.D);
        end
        
        function [ B ] = real( A )
            B = A;
            B.Diag = real(B.Diag);
            B.D = real(B.D);
        end
        
        function [ B ] = imag( A )
            B = A;
            B.D = imag(B.D);
            B.Diag = imag(B.Diag);
        end
        
        function [ B ] = full( A )
            if A.N > 5000
                error('FULL: Matrix too large; threshold set at 5000x5000');
            end
            B = A * eye(size(A),'double');
        end
        
        function [ varargout ] = size( A, dim )
            if nargin < 2
                siz = [A.N, A.N];
            else
                if dim == 1 || dim == 2
                    siz = A.N;
                else
                    if dim == round(dim) && dim > 0
                        siz = 1;
                    else
                        error('dim must be a positive integer');
                    end
                end
                if nargout > 1
                    error('Too many output arguments');
                end
            end
            if nargout <= 1
                varargout{1} = siz;
            elseif nargout >= 2
                varargout{1} = siz(1);
                varargout{2} = siz(2);
                for ii = 3:nargout
                    varargout{ii} = 1;
                end
            end
        end
        
        function [ len ] = length( A )
            len = A.N;
        end
        
        function [ num ] = numel( A )
            num = A.N^2;
        end
        
        function [ out ] = norm( A, p )
            if nargin < 2
                p = 2;
            end
            
            if isa(p,'char') && strcmpi(p,'fro')
                out = sqrt( sum(abs(A.Diag(:)).^2) + A.N * sum(offdiagonals(A).^2) );
            elseif isnumeric(p) && isscalar(p) && (p == 1 || p == inf)
                % 1-norm is same as infinity-norm for symmetric matrices
                out = maxabs(diag(A)) + sum(offdiagonals(A));
            else
                error('Only 1- and infinity-norms are implemented for BlochTorreyOp''s');
            end
        end
        
        function [ bool ] = isscalar( A )
            bool = (A.N == 1);
        end
        
        function [ bool ] = isnumeric( A )
            bool = true;
        end
        
        function [ bool ] = isfloat( A )
            bool = true;
        end
        
        function [ bool ] = isdiag( A )
            bool = (A.D == 0);
        end
        
        function [ bool ] = isreal( A )
            bool = isreal(A.Diag) && isreal(A.D);
        end
        
        function [ B ] = conj( A )
            B = A;
            B.Diag = conj(B.Diag);
            B.D = conj(B.D);
        end
        
        function [ y ] = complex( A, B )
            if ~(isreal(A) && isreal(B))
                error('Both inputs to COMPLEX must be real');
            end
            y = A + 1i .* B;
        end
        
        function [ bool ] = issymmetric( A )
            bool = true;
        end
        
        function [ bool ] = ishermitian( A )
            bool = isreal(A.D) && isreal(A.Diag);
        end
        
        function [ bool ] = ishandle( A )
            bool = true;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CUSTOM METHODS:
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ bool ] = iscompatible( A, B ) %ISCOMPATIBLE
            AIsBTOp = isa(A, 'BlochTorreyOp');
            BIsBTOp = isa(B, 'BlochTorreyOp');
            
            if AIsBTOp && BIsBTOp
                bool = ( isequal(A.gsize, B.gsize) && ... %must operate on same grid
                    isequal(A.gdims, B.gdims) );     %grids must have same physical dimensions
            else
                error('ISCOMPATIBLE(A,B) is for comparing two BlochTorreyOp''s A and B');
            end
        end
        
        function [ bool ] = isisotropic( A ) %ISISOTROPIC
            bool = ( norm(diff(A.h)) < 2*eps(class(A.h)) );
        end
        
        function [ out ] = offdiagonals( A ) %OFFDIAGONALS
            %Returns list of 6 off-diagonal elements, in no particular
            %order. These 6 elements are the same for any row or column.
            out = A.D ./ [A.h, A.h].^2;
        end
        
        function [ c, mv ] = normAm( A, m, checkpos ) %jd
            %NORMAM   Estimate of 1-norm of power of matrix.
            %   NORMAM(A,m) estimates norm(A^m,1).
            %   If A has nonnegative elements the estimate is exact.
            %   [C,MV] = NORMAM(A,m) returns the estimate C and the number MV of
            %   matrix-vector products computed involving A or A^*.
            
            %upper bound for norm(A^m): typically very accurate in practice due
            %to A being complex symmetric & diagonally dominant, and MUCH
            %faster than approximating it via normest1.
            c = norm(A,1)^m;
            mv = 0;
            
            %   Reference: A. H. Al-Mohy and N. J. Higham, A New Scaling and Squaring
            %   Algorithm for the Matrix Exponential, SIAM J. Matrix Anal. Appl. 31(3):
            %   970-989, 2009.
            
            %   Awad H. Al-Mohy and Nicholas J. Higham, September 7, 2010.
            %
            %if nargin < 3 %jd
            %    checkpos = false; %default for this class (it's normally complex). jd
            %end
            %
            %t = 1; % Number of columns used by NORMEST1.
            %if checkpos && isequal(A,abs(A)) %Optionally skip check. jd
            %    e = ones(length(A),1);
            %    for j=1:m         % for positive matrices only
            %        e = A'*e;
            %    end
            %    c = norm(e,inf);
            %    mv = m;
            %else
            %    [c,v,w,it] = normest1(@afun_power, t, [], A.h, A.D, A.Diag, A.gsize, m);
            %    mv = it(2)*t*m;
            %end
            %
            %function Z = afun_power(flag,X,h,D,Diag,gsize,m)
            %    %AFUN_POWER  Function to evaluate matrix products needed by NORMEST1.
            %
            %    if isequal(flag,'dim')
            %        Z = prod(gsize);
            %    elseif isequal(flag,'real')
            %        Z = isreal(Diag);
            %    else
            %        Z = BlochTorreyAction(X, h, D, Diag, gsize, m, flag);
            %    end
            %end
            
        end
        
        function [ out ] = getBuffer(A)
            out = A.buffer;
        end
        
        function [ out ] = getState(A)
            out = A.state;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % HIDDEN METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Hidden)
        
        function disp(A)
            classtype = class(A.Diag);
            cplxstr = ''; if ~isreal(A.Diag); cplxstr = 'complex '; end
            fprintf('%dx%d BlochTorreyOp array with properties:\n\n',A.N,A.N);
            fprintf('   gsize: %s\n', mat2str(A.gsize));
            fprintf('   gdims: %s\n', mat2str(A.gdims));
            fprintf('       N: %d\n', A.N);
            fprintf('       h: %s\n', mat2str(A.h));
            fprintf('       D: %s\n', num2str(A.D));
            fprintf('    Diag: %s %s\n', mat2str(A.gsize), [cplxstr, classtype]);
            fprintf('   Gamma: %s %s\n', mat2str(A.gsize), [cplxstr, classtype]);
            fprintf('   R2map: %s %s\n', mat2str(A.gsize), classtype);
            fprintf('  dOmega: %s %s\n', mat2str(A.gsize), classtype);
            fprintf('\n');
        end
        
        function B = zerosLike( A, varargin )
            if nargin == 1
                % For zeros('like',obj)
                B = A;
                B.D = 0;
                B.Diag = 0;
            elseif  any([varargin{:}] <= 0)
                % For zeros with any dimension <= 0
                error('Dimensions <= 0, and empty method is not implemented.');
            else
                % For zeros(m,n,...,'like',obj)
                if ~isequal([varargin{:}], size(A))
                    error('ZEROS: must use zeros(size(A),''like'',A)');
                end
                B = BlochTorreyOp(0,0,A.gsize,A.gdims);
            end
        end
        
        function B = eyeLike( A, varargin )
            if nargin == 1
                % For eye('like',obj)
                B = A;
                B.D = 0;
                B.Diag = 1;
            elseif any([varargin{:}] <= 0)
                % For eye with any dimension <= 0
                error('Dimensions <= 0, and empty method is not implemented.');
            else
                % For eye(m,n,...,'like',obj)
                if length(varargin) == 1 && isscalar(varargin{1})
                    varargin{1} = [varargin{1}, varargin{1}];
                end
                if ~isequal([varargin{:}], size(A))
                    error('ZEROS: must use zeros(size(A),''like'',A)');
                end
                B = BlochTorreyOp(-1,0,A.gsize,A.gdims);
            end
        end
        
    end
    
end

