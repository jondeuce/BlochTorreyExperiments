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
        N     % Total number of elements on grid, i.e. `prod(gsize)`
        h     % Physical distance between elements, e.g. `gdims./gsize` = [5.8594,5.8594,5.8594] um
    end

    properties ( GetAccess = public, SetAccess = private )
        D      % Diffusion coefficient, e.g. 3037 um^2/s
        mask   % Vasculature mask of size `gsize`, or defaults to []
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

        function [ A ] = BlochTorreyOp( Buffer, Dcoeff, GridSize, GridDims, isdiag, mask )
            % [ A ] = BlochTorreyOp( Gamma, Dcoeff, GridSize, GridDims )
            %   INPUTS:
            %       Buffer:     Complex array which may represent Gamma = R2 + i*dw, or the diagonal of the numeral BT operator
            %       Dcoeff:     Diffusion coefficient [um^2/s == 1000 * mm^2/ms], e.g. water ~ 3037 um^2/s
            %       GridSize:   Grid size, e.g. [512,512,512]
            %       GridDims:   Grid dimensions, e.g. [3000,3000,3000] (in um)

            if nargin < 6; mask = logical([]); end % default is mask = []
            if nargin < 5; isdiag = false; end % default is Buffer = Gamma

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

            if ~BlochTorreyOp.is_isotropic(GridDims./GridSize)
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
            A.mask   = mask;
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
                diag = calculate_diagonal(A.D, A.buffer, A.h, A.gsize, A.mask);
            end
        end
        function gamma = get.Gamma(A)
            if A.state == BlochTorreyOp.DiagState
                gamma = calculate_gamma(A.D, A.buffer, A.h, A.gsize, A.mask);
            else % BlochTorreyOp.GammaState
                gamma = A.buffer;
            end
        end

        function A = set.Diag(A, diag)
            A.buffer = diag;
            A.state  = BlochTorreyOp.DiagState;
        end
        function A = set.Gamma(A, gamma)
            A.buffer = gamma;
            A.state  = BlochTorreyOp.GammaState;
        end

        function A = setbuffer(A, State)
            if isequal(State, A.state)
                % already in state State; do nothing
                return
            end
            switch State
                case BlochTorreyOp.DiagState
                    % in GammaState; switch to DiagState
                    A.buffer = calculate_diagonal(A.D, A.buffer, A.h, A.gsize, A.mask);
                    A.state  = BlochTorreyOp.DiagState;
                case BlochTorreyOp.GammaState
                    % in DiagState; switch to GammaState
                    A.buffer = calculate_gamma(A.D, A.buffer, A.h, A.gsize, A.mask);
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
            A.Gamma = complex(r2, A.dOmega);
        end
        function A = set.dOmega(A,dw)
            A.Gamma = complex(A.R2map, dw);
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
                        % A is simply a diagonal matrix, with negative-Gamma on the diagonal
                        if isscalar(A.buffer)
                            diag = A.Diag; % cheap, since buffer is scalar
                            tol  = 5*eps(class(diag));
                            if abs(diag) <= tol
                                y =  zeros(size(x),'like',x);
                            elseif abs(diag - 1) <= tol
                                y =  x;
                            elseif abs(diag + 1) <= tol
                                y = -x;
                            else
                                y = diag .* x;
                            end
                        else
                            diag = A.Diag; % equally cheap to taking -A.Gamma when A.D == 0
                            if size(A,2) == numel(x)
                                y = reshape(diag, size(x)) .* x;
                            else
                                ncols = round(numel(x)/size(A,2));
                                if ncols * size(A,2) ~= numel(x)
                                    error('size(x) must be either the (repeated-)grid size or (repeated-)flattened size');
                                end
                                if ismatrix(x)
                                    % x is a (possibly repeated) matrix;
                                    % multiply x across rows by diag
                                    y = bsxfun(@times,diag(:),x);
                                else
                                    % x is a (possibly repeated) 3D array;
                                    % multiply x along dimensions 1:3 by diag
                                    sizx = size(x);
                                    y = bsxfun(@times,reshape(diag,sizx(1:3)),x);
                                end
                            end
                        end
                    else
                        iters = 1; % one iteration, i.e. x->A*x
                        istrans = false; % regular application, not A'
                        isdiag = (A.state == BlochTorreyOp.DiagState);
                        y = BlochTorreyAction(x, A.h, A.D, A.buffer, A.gsize, iters, istrans, isdiag, A.mask);
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
                    if ismatrix(A)
                        % A is the vector or matrix being acted on
                        if isrealx && isrealA
                            %(x' * A')' = (x.' * A.').'
                            y = x.' * A.';
                        elseif  isrealx && ~isrealA
                            %(x' * A')' = (x.' * A')'
                            y = (x.' * A')';
                        elseif ~isrealx &&  isrealA
                            %(x' * A')' = (x' * A.')'
                            y = (x' * A.')';
                        else % ~isrealx && ~isrealA
                            %(x' * A')'
                            y = (x' * A')';
                        end
                    else
                        % In this case, we interpret the 3D array A as a
                        % vector, and do conjugation manually
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
                    y        = A;
                    y.buffer = y.buffer .* x;
                    y.D      = y.D .* x;
                else
                    error('Only scalar multiplication is allowed on RHS of BlochTorreyOp');
                end
            elseif ~AIsBTOp && xIsBTOp
                if isscalar(A) && isnumeric(A)
                    y        = x;
                    y.buffer = A .* y.buffer;
                    y.D      = A .* y.D;
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

                if A.state == B.state
                    y = A;
                    y.buffer = y.buffer + B.buffer;
                elseif A.state == BlochTorreyOp.DiagState
                    y = A;
                    y.buffer = y.buffer + B.Diag;
                else
                    y = B;
                    y.buffer = y.buffer + A.Diag;
                end
                y.D = y.D + B.D;

            elseif  AIsBTOp && ~BIsBTOp
                if isequal(size(A),size(B)) && isdiag(B)
                    y = A;
                    if A.state == BlochTorreyOp.DiagState
                        if isscalar(y.buffer)
                            y.buffer = y.buffer + reshape(full(diag(B)),A.gsize);
                        else
                            y.buffer = y.buffer + reshape(full(diag(B)),size(y.buffer));
                        end
                    else
                        if isscalar(y.buffer)
                            y.buffer = y.buffer - reshape(full(diag(B)),A.gsize);
                        else
                            y.buffer = y.buffer - reshape(full(diag(B)),size(y.buffer));
                        end
                    end
                else
                    error('PLUS: second argument must be a BlochTorreyOp or a diagonal matrix.');
                end
            elseif ~AIsBTOp &&  BIsBTOp
                if isequal(size(A),size(B)) && isdiag(A)
                    y = B;
                    if A.state == BlochTorreyOp.DiagState
                        if isscalar(y.buffer)
                            y.buffer = y.buffer + reshape(full(diag(A)),A.gsize);
                        else
                            y.buffer = y.buffer + reshape(full(diag(A)),size(y.buffer));
                        end
                    else
                        if isscalar(y.buffer)
                            y.buffer = y.buffer - reshape(full(diag(A)),A.gsize);
                        else
                            y.buffer = y.buffer - reshape(full(diag(A)),size(y.buffer));
                        end
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
            isdiagstate = (A.state == BlochTorreyOp.DiagState);
            B = BlochTorreyOp( -A.buffer, -A.D, A.gsize, A.gdims, isdiagstate );
        end

        function [ B ] = transpose( A )
            % it's symmetric; return input
            B = A;
        end

        function [ B ] = ctranspose( A )
            if isreal(A)
                B = A;
            else
                isdiagstate = (A.state == BlochTorreyOp.DiagState);
                B = BlochTorreyOp( conj(A.buffer), conj(A.D), A.gsize, A.gdims, isdiagstate );
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
            % matrix trace
            if isscalar(A.buffer)
                Tr = A.buffer * A.N;
            else
                Tr = sumall(A.buffer);
            end

            if A.state == BlochTorreyOp.GammaState
                if isscalar(A.D)
                    Tr = (-6*A.D/mean(A.h)^2) * A.N - Tr;
                else
                    Tr = (-6/mean(A.h)^2)*sumall(A.D) - Tr;
                end
            end
        end

        function [ B ] = abs( A )
            % need to take full abs(A.Diag); no linearity
            B = BlochTorreyOp( abs(A.Diag), abs(A.D), A.gsize, A.gdims, true ); % forward gradient/backward divergence
        end

        function [ B ] = real( A )
            isdiagstate = (A.state == BlochTorreyOp.DiagState);
            B = BlochTorreyOp( real(A.buffer), real(A.D), A.gsize, A.gdims, isdiagstate );
        end

        function [ B ] = imag( A )
            isdiagstate = (A.state == BlochTorreyOp.DiagState);
            B = BlochTorreyOp( imag(A.buffer), imag(A.D), A.gsize, A.gdims, isdiagstate );
        end

        function [ B ] = conj( A )
            isdiagstate = (A.state == BlochTorreyOp.DiagState);
            B = BlochTorreyOp( conj(A.buffer), conj(A.D), A.gsize, A.gdims, isdiagstate );
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

            hh = mean(A.h);

            if isa(p,'char') && strcmpi(p,'fro')
                diag = A.Diag;
                if isscalar(A.D)
                    if isscalar(diag)
                        out = sqrt( A.N * abs2(diag) + A.N * sum(abs2(offdiagonals(A))) );
                    else
                        out = sqrt( sumall(abs2(diag)) + A.N * sum(abs2(offdiagonals(A))) );
                    end
                else
                    % % forward gradient/backward divergence
                    % out = sqrt(sumall( abs2(diag) + (6/hh^4)*abs2(A.D) ));
                    
                    % flux difference
                    K = (0.5/mean(A.h)^2);
                    tmp = abs2(diag);
                    tmp = tmp + abs2(SevenPointStencil(A.D, K * [1,1,0,0,0,0,0], A.gsize, 1));
                    tmp = tmp + abs2(SevenPointStencil(A.D, K * [1,0,1,0,0,0,0], A.gsize, 1));
                    tmp = tmp + abs2(SevenPointStencil(A.D, K * [1,0,0,1,0,0,0], A.gsize, 1));
                    tmp = tmp + abs2(SevenPointStencil(A.D, K * [1,0,0,0,1,0,0], A.gsize, 1));
                    tmp = tmp + abs2(SevenPointStencil(A.D, K * [1,0,0,0,0,1,0], A.gsize, 1));
                    tmp = tmp + abs2(SevenPointStencil(A.D, K * [1,0,0,0,0,0,1], A.gsize, 1));
                    out = sqrt(sumall(tmp));
                end
            elseif isnumeric(p) && isscalar(p) && (p == 1 || p == inf)
                % 1-norm is same as infinity-norm for symmetric matrices
                diag = A.Diag;
                if isscalar(A.D)
                    out = infnorm(diag) + sum(abs(offdiagonals(A)));
                else
                    % % forward gradient/backward divergence:
                    % kern = (1/mean(A.h)^2) * [3,1,0,1,0,1,0];
                    % out = maximum(abs(diag) + SevenPointStencil(abs(A.D), kern, A.gsize, 1));
                    
                    % flux difference
                    K = (0.5/mean(A.h)^2);
                    tmp = abs(diag);
                    tmp = tmp + abs(SevenPointStencil(A.D, K * [1,1,0,0,0,0,0], A.gsize, 1));
                    tmp = tmp + abs(SevenPointStencil(A.D, K * [1,0,1,0,0,0,0], A.gsize, 1));
                    tmp = tmp + abs(SevenPointStencil(A.D, K * [1,0,0,1,0,0,0], A.gsize, 1));
                    tmp = tmp + abs(SevenPointStencil(A.D, K * [1,0,0,0,1,0,0], A.gsize, 1));
                    tmp = tmp + abs(SevenPointStencil(A.D, K * [1,0,0,0,0,1,0], A.gsize, 1));
                    tmp = tmp + abs(SevenPointStencil(A.D, K * [1,0,0,0,0,0,1], A.gsize, 1));
                    out = maximum(tmp);
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
            bool = isreal(A.D) && isreal(A.buffer);
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
            bool = isreal(A.D) && isreal(A.buffer);
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
            bool = BlochTorreyOp.is_isotropic(A.h);
        end

        function [ out ] = offdiagonals( A ) %OFFDIAGONALS
            %Returns list of 6 off-diagonal elements, in no particular
            %order. For constant isotropic diffusion, the 6 elements
            %returned are the same for any row or column.
            %For variable isotropic diffusion, the 6 elements for each row
            %are returned (by symmetry, they are the same as the
            %offdiagonal elements for the corresponding column).
            out = calculate_offdiagonals(A.D, A.h, A.mask);
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
            classtype = class(A.buffer);
            cplxstr = ''; if ~isreal(A.buffer); cplxstr = 'complex '; end
            Dcplxstr = ''; if ~isreal(A.D); Dcplxstr = 'complex '; end
            fprintf('%dx%d BlochTorreyOp array with properties:\n\n',A.N,A.N);
            fprintf('   gsize: %s\n', mat2str(A.gsize));
            fprintf('   gdims: %s\n', mat2str(A.gdims));
            fprintf('       N: %d\n', A.N);
            fprintf('       h: %s\n', mat2str(A.h));
            if isscalar(A.D)
            fprintf('       D: %s\n', num2str(A.D));
            else
            fprintf('       D: %s %s\n', mat2str(A.gsize), [Dcplxstr, classtype]);
            end
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
        % Testing script
        all_tests_passed = test(Gsize)

        % Check if grid size is isotropic
        function b = is_isotropic(h)
            b = isscalar(h) || (max(abs(diff(h))) <= 10*eps(max(abs(h(:)))));
        end

    end

end

function y = sumall(x)
% faster and more accurate than sum(x(:))
y = sum(sum(sum(sum(x,1),2),3),4);
end

function out = calculate_offdiagonals(D, h, mask)

if ~BlochTorreyOp.is_isotropic(h)
    error('h must be isotropic for offdiagonals calculation');
end
h = mean(h(:));

if isscalar(D)
    out = (D/h^2) * ones(1,6);
else
    % out = [ vec(D), vec(D), vec(D), vec(circshift(D,1,1)), vec(circshift(D,1,2)), vec(circshift(D,1,3)) ]; % forward grad/backward div
    out = [ vec(D + circshift(D, 1, 1)), vec(D + circshift(D, 1, 2)), vec(D + circshift(D, 1, 3)), ...
            vec(D + circshift(D,-1, 1)), vec(D + circshift(D,-1, 2)), vec(D + circshift(D,-1, 3)) ]; % flux difference
end

end

function Diagonal = calculate_diagonal(D, Gamma, h, gsize, mask)

if ~BlochTorreyOp.is_isotropic(h)
    error('h must be isotropic for diagonal calculation');
end
h = mean(h(:));

if isscalar(D)
    if isequal(D, 0)
        Diagonal = -Gamma;
    else
        Diagonal = (-6*D/h^2) - Gamma;
    end
else
    % % Backward divergence/forward gradient div(D*grad(x))
    % kern = (-1/h^2) .* [3,1,0,1,0,1,0];
    % Diagonal = SevenPointStencil(D, kern, gsize, 1) - Gamma;

    % % Slower version of above
    % Diagonal = (-1/h^2)*(3*D + circshift(D,1,1) + circshift(D,1,2) + circshift(D,1,3)) - Gamma;

    % % Symmetrized D*lap(x)-dot(grad(D),grad(x))
    % Diagonal = 0.5*Laplacian(D,h,size(D),1) - (6/h^2)*D - Gamma;

    % Flux difference for div(D*grad(x))
    kern = (-0.5/h^2) .* [6,1,1,1,1,1,1];
    Diagonal = SevenPointStencil(D, kern, gsize, 1) - Gamma;
end

end

function Gamma = calculate_gamma(D, Diagonal, h, gsize, mask)

% By linearity, since Diagonal == L - Gamma for some generic diagonal L, we
% similarly have Gamma == L - Diagonal. Therefore, just reuse the
% calculate_diagonal function
Gamma = calculate_diagonal(D, Diagonal, h, gsize, mask);

end
