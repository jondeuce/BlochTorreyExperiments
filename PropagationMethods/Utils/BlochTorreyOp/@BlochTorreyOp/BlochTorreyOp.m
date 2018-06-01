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
        
        function [ A ] = BlochTorreyOp( Buffer, Dcoeff, GridSize, GridDims, isdiag )
            % [ A ] = BlochTorreyOp( Gamma, Dcoeff, GridSize, GridDims )
            %   INPUTS:
            %       Buffer:     Complex array which may represent Gamma = R2 + i*dw, or the diagonal of the numeral BT operator
            %       Dcoeff:     Diffusion coefficient [um^2/s == 1000 * mm^2/ms], e.g. water ~ 3037 um^2/s
            %       GridSize:   Grid size, e.g. [512,512,512]
            %       GridDims:   Grid dimensions, e.g. [3000,3000,3000] (in um)
            
            if nargin < 5
                isdiag = false; %default is Buffer = Gamma
            end
            
            if ~(isscalar(Buffer) || isequal(numel(Buffer), prod(GridSize)))
                error('Gamma must be scalar or have the same number of elements as the grid');
            end
            
            if ~(isscalar(Dcoeff) || isequal(numel(Dcoeff), prod(GridSize)))
                error('Diffusion coefficient must be scalar or have the same number of elements as the grid');
            end
            
            if ~isequal(numel(GridSize),3) || ~isequal(numel(GridDims),3)
                error('gsize and gdims must be 3-element vectors, corresponding to a 3D grid');
            end
            GridSize = GridSize(:).';
            GridDims = GridDims(:).';
            
            if ~is_isotropic(GridDims./GridSize)
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
            A.buffer = Buffer;
            if isdiag
                A.state  = BlochTorreyOp.DiagState;
            else
                A.state  = BlochTorreyOp.GammaState;
            end
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
                diag = calculate_diagonal(A.D,A.buffer,A.h);
            end
        end
        function gamma = get.Gamma(A)
            if A.state == BlochTorreyOp.DiagState
                gamma = calculate_gamma(A.D,A.buffer,A.h);
            else % BlochTorreyOp.GammaState
                gamma = A.buffer;
            end
        end
        
        function A = set.Diag(A,diag)
            A.buffer = diag;
            A.state  = BlochTorreyOp.DiagState;
        end
        function A = set.Gamma(A,gamma)
            A.buffer = gamma;
            A.state  = BlochTorreyOp.GammaState;
        end
        
        function A = switchbuffer(A,State)
            if isequal(State, A.state)
                return % already in state State, do nothing
            end
            switch State
                case BlochTorreyOp.DiagState % in GammaState; switch to DiagState
                    A.buffer = calculate_diagonal(A.D,A.buffer,A.h);
                    A.state  = BlochTorreyOp.DiagState;
                case BlochTorreyOp.GammaState % in DiagState; switch to GammaState
                    A.buffer = calculate_gamma(A.D,A.buffer,A.h);
                    A.state  = BlochTorreyOp.GammaState;
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
                error('Composition of BlochTorreyOp''s is not supported');
            elseif  AIsBTOp && ~xIsBTOp
                if isscalar(x) && isnumeric(x)
                    y = times( A, x ); %A should act like a MATRIX, not an OPERATOR
                else
                    if isequal(A.D, 0)
                        % A is simply a diagonal matrix, with minus-Gamma
                        % on the diagonal
                        if isscalar(A.Diag)
                            if abs(A.Diag - 1) <= 5*eps(class(A.Diag))
                                y = x;
                            else
                                y = A.Diag .* x;
                            end
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
                                    y = bsxfun(@times,reshape(A.Diag,sizx(1:3)),x);
                                else
                                    y = bsxfun(@times,A.Diag(:),x);
                                end
                            end
                        end
                    else
                        if isscalar(A.D)
                            y = BlochTorreyAction(x, A.h, A.D, A.Diag, A.gsize, 1, false, true);
                        else
                            y = BlochTorreyAction(x, A.h, A.D, A.Gamma, A.gsize, 1, false, false);
                        end
                    end
                end
            elseif ~AIsBTOp && xIsBTOp
                if isscalar(A) && isnumeric(A)
                    y = times( A, x );
                else
                    % A*x = (x' * A')' and variants (where x is the BT
                    % operator) can be reimplemented using the mtimes
                    % branch above, assuming that (c)transpose is known for
                    % BT operators
                    isrealx = isreal(x);
                    isrealA = isreal(A);
                    if isvector(x)
                        if isrealx && isrealA
                            %(x' * A')' = (x.' * A.').'
                            y = x.' * A;
                        elseif  isrealx && ~isrealA
                            %(x' * A')' = (x.' * A')'
                            y = conj(x.' * conj(A));
                        elseif ~isrealx &&  isrealA
                            %(x' * A')' = (x' * A.')'
                            y = conj(x' * A);
                        else % ~isrealx && ~isrealA
                            %(x' * A')'
                            y = conj(x' * conj(A));
                        end
                    else
                        if isrealx && isrealA
                            %(x' * A')' = (x.' * A.').'
                            y = x.' * A; %y = reshape( x * A(:), size(A) );
                        elseif  isrealx && ~isrealA
                            %(x' * A')' = (x.' * A')'
                            y = conj(x.' * conj(A)); %y = reshape( conj(x * conj(A(:))), size(A) );
                        elseif ~isrealx &&  isrealA
                            %(x' * A')' = (x' * A.')'
                            y = conj(x' * A); %y = reshape( conj(x' * A(:)), size(A) );
                        else % ~isrealx && ~isrealA
                            %(x' * A')'
                            y = conj(x' * conj(A)); %y = reshape( conj(x' * conj(A(:))), size(A) );
                        end
                    end
                end
            end
        end
        
        function [ y ] = times( A, x )
            AIsBTOp = isa(A, 'BlochTorreyOp');
            xIsBTOp = isa(x, 'BlochTorreyOp');
            
            if AIsBTOp && xIsBTOp
                % if iscompatible( A, x ) && isisotropic( A )
                %     %isotropic grid is required, else can't have single equivalent D
                %     y = A;
                %     if ~isequal( size(y.Diag), size(x.Diag) ) && ~( isscalar(y.Diag) || isscalar(x.Diag) )
                %         % If one is scalar, no need to reshape; just broadcast and multiply
                %         y.Diag = reshape(y.Diag, size(x.Diag));
                %     end
                %     y.Diag = y.Diag .* x.Diag; %simply multiply diagonals
                %     y.D    = (y.D / y.h(1))^2; %off-diags are each D/h^2; (D/h^2)^2 = (D^2/h^2)/h^2 = D_new/h^2
                % else
                %     error('Multiplying BlochTorreyOp''s is only supported for isotropic grids of the same size, and physical dimensions');
                % end
                error('Multiplying BlochTorreyOp''s is not supported.');
            elseif AIsBTOp && ~xIsBTOp
                if isscalar(x) && isnumeric(x)
                    y      = A;
                    y.Diag = y.Diag .* x;
                    y.D    = y.D .* x;
                else
                    error('Only scalar multiplication is allowed on RHS of BlochTorreyOp');
                end
            elseif ~AIsBTOp && xIsBTOp
                if isscalar(A) && isnumeric(A)
                    y      = x;
                    y.Diag = A .* y.Diag;
                    y.D    = A .* y.D;
                else
                    error('Only scalar multiplication is allowed on LHS of BlochTorreyOp');
                end
            end
        end
        
        function [ y ] = plus( A, B )
            AIsBTOp = isa(A, 'BlochTorreyOp');
            BIsBTOp = isa(B, 'BlochTorreyOp');
            
            if AIsBTOp && BIsBTOp
                if ~iscompatible(A,B)
                    error('PLUS: Dimension mismatch');
                end
                y = A;
                y.Diag = y.Diag + B.Diag;
                y.D    = y.D    + B.D;
            elseif  AIsBTOp && ~BIsBTOp
                if isequal(size(A),size(B)) && isdiag(B)
                    y = A;
                    if isscalar(y.Diag)
                        y.Diag = y.Diag + reshape(full(diag(B)),A.gsize);
                    else
                        y.Diag = y.Diag + reshape(full(diag(B)),size(y.Diag));
                    end
                else
                    error('PLUS: second argument must be a BlochTorreyOp or a diagonal matrix.');
                end
            elseif ~AIsBTOp &&  BIsBTOp
                if isequal(size(A),size(B)) && isdiag(A)
                    y = B;
                    if isscalar(y.Diag)
                        y.Diag = y.Diag + reshape(full(diag(A)),A.gsize);
                    else
                        y.Diag = y.Diag + reshape(full(diag(A)),size(y.Diag));
                    end
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
            B.Diag = -B.Diag;
            B.D = -B.D;
        end
        
        function [ B ] = transpose( A )
            % it's symmetric; return input
            %TODO symmetric for non-constant D?
            B = A;
        end
        
        function [ B ] = ctranspose( A )
            B = A;
            if ~isreal(B)
                B.Diag = conj(B.Diag);
                B.D = conj(B.D);
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
                Tr = sum(sum(sum(A.Diag,1),2),3);
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
            B.Diag = imag(B.Diag);
            B.D = imag(B.D);
        end
        
        function [ B ] = full( A, thresh )
            if nargin < 2; thresh = 5000; end
            if A.N > thresh
                error('FULL: Matrix too large; threshold set at size(A) = %dx%d', thresh, thresh);
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
                if isscalar(A.D)
                    if isscalar(A.Diag)
                        out = sqrt( A.N * abs(A.Diag).^2 + A.N * sum(offdiagonals(A).^2) );
                    else
                        out = sqrt( sum(abs(A.Diag(:)).^2) + A.N * sum(offdiagonals(A).^2) );
                    end
                else
                    %TODO
                    error('Frobenius norm not implemented for non-scalar D.');
                end
            elseif isnumeric(p) && isscalar(p) && (p == 1 || p == inf)
                if isscalar(A.D)
                    % 1-norm is same as infinity-norm for symmetric matrices
                    out = maxabs(diag(A)) + sum(offdiagonals(A));
                else
                    %TODO
                    error('1-norm and infinity-norm not implemented for non-scalar D.');
                end
            else
                error('Only Frobenius-, 1-, and infinity-norms are implemented for BlochTorreyOp''s');
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
            bool = isequal(A.D, 0);
        end
        
        function [ bool ] = isreal( A )
            bool = isreal(A.D) && isreal(A.Diag);
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
            %TODO is it for non-scalar D?
            bool = true;
        end
        
        function [ bool ] = ishermitian( A )
            %TODO is it for non-scalar D?
            bool = isreal(A.D) && isreal(A.Diag);
        end
        
        function [ bool ] = ishandle( A )
            bool = false;
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
            bool = ( max(abs(diff(A.h))) <= 5*eps(max(A.h)) );
        end
        
        function [ out ] = offdiagonals( A ) %OFFDIAGONALS
            %Returns list of 6 off-diagonal elements, in no particular
            %order. These 6 elements are the same for any row or column.
            out = calculate_offdiagonals(A.D,A.h);
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
                B = BlochTorreyOp(0,0,A.gsize,A.gdims);
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
                B = BlochTorreyOp(-1,0,A.gsize,A.gdims);
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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STATIC METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Static = true)
        test
    end
    
end

function b = is_isotropic(h)
b = isscalar(h) || (max(abs(diff(h))) <= 10*eps(max(h)));
end

function f = calculate_offdiagonals(D,h)

if ~is_isotropic(h)
    error('h must be isotropic for offdiagonals calculation');
end
h = mean(h(:));

if isscalar(D)
    f = (D/h^2) * ones(1,6);
else
    %TODO
    error('offdiagonals(A) not implemented for non-scalar D.');
    f = 0.5*Laplacian(D,h,size(D),1) - (6/h^2)*D - Gamma;
end

end

function Diagonal = calculate_diagonal(D,Gamma,h)

if ~is_isotropic(h)
    error('h must be isotropic for diagonal calculation');
end
h = mean(h(:));

if isscalar(D)
    Diagonal = (-6*D/h^2) - Gamma;
else
    Diagonal = 0.5*Laplacian(D,h,size(D),1) - (6/h^2)*D - Gamma;
end

end

function Gamma = calculate_gamma(D,Diagonal,h)

if ~is_isotropic(h)
    error('h must be isotropic for diagonal calculation');
end
h = mean(h(:));

if isscalar(D)
    Gamma = (-6*D/h^2) - Diagonal;
else
    Gamma = 0.5*Laplacian(D,h,size(D),1) - (6/h^2)*D - Diagonal;
end

end

