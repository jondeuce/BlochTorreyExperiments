classdef matfun
    %MATFUN Class to represent functions of the form f(x) = A*x.
    % 
    % Note of use:
    %   - In general, f(x) may be any function that acts linearly on the
    %     argument x, and as such f(x) will be interpreted as a matrix, A.
    %     f may be created, for example, by f = matfun(@(x)A*x,size(A)).
    %   - It is NOT assumed anywhere that x is a vector.
    %	- If A has size [m,n], then it must be that numel(x) = n for A*x to
    %	  be valid, or numel(y) = m for y*A to be valid.
    %   - For y*A to be valid for a vector y, ctranspose must be be defined
    %	  for A and be valid for y, as y*A is implimented as y*A = (A'*y')'
    %	- There are two options for left multiplication by a matrix B:
    %       1. Ensure that the size(B,2) == size(A,1), AND numel(B) ~=
    %          size(A,2). This second condition is to ensure that B will
    %          not be interpreted as a vector. If you wish to multiply from
    %          the left by a row vector and return a matfun, use option 2.
    %       2. Create a matfun g = matfun(@(x)B*x,size(B)) and evaluate the
    %          product g*f. A matfun object will be returned.
    %   - The size of A may be 'infinite' in dimension 1, 2, or both. This
    %     is interpreted as a function that acts linearly a vector of any
    %     length in the corresponding dimension.
    
    properties (Access = private)
        
        f       % Function representing f(x) = A*x
        m       % size(A,1)
        n       % size(A,2)
        bops	% Binary operations on the matrix A
        uops	% Unary operations on the matrix A
        gops	% General operations on A which do not necessarily return matfuns
        
    end
    
    methods
        
        %==================================================================
        % Constructor
        %==================================================================
        function F = matfun( f, Asize, varargin )
            
            if nargin < 2, error('Must provide A size'); end
            if nargin < 1, error('Must provide A matrix'); end
            
            if ~isa(f,'function_handle'), error('f must be a function handle'); end
            if ~(numel(Asize)==2), error('Matrix must be 2 dimensional'); end
            
            F.f     =   f;
            F.m     =   Asize(1);
            F.n     =   Asize(2);
            
            F.bops	=   matfun.default_bops;
            F.uops	=   matfun.default_uops;
            F.gops	=   matfun.default_gops;
            F       =   set_default_ops(F);
            F       =   setop(F,varargin{:});
            
        end
        
        %==================================================================
        % Matrix multiplication
        %==================================================================
        function b = mtimes(F,x)
            
            isF	=   isa(F,'matfun');
            isx	=   isa(x,'matfun');
            err	=   @(Fsiz,xsiz) error( ['Matrix dimensions must agree for A*B.'...
                ' A has size [%d,%d] while B has size [%d,%d].\n'],Fsiz,xsiz);
            
            if isF && ~isx
                if isinf(size(F,2))
                    b	=   F.f(x); % any rhs vector is valid
                else
                    if size(F,2) == numel(x)
                        b	=   F.f(x); % interpret x as a vector and eval
                    elseif size(F,2) == size(x,1)
                        x	=   matfun(@(y)x*y,size(x)); % x is a matrix
                        b	=   F*x; % Compose matrix mult. of F and x
                    else
                        err(size(F),[numel(x),1]);
                    end
                end
            elseif isF && isx
                if isinf(size(F,2)) || isinf(size(x,1)) || (size(F,2)==size(x,1))
                    b	=   matfun( @(y) F.f(x.f(y)), [size(F,1),size(x,2)] );
                else
                    err(size(F),size(x));
                end
            else % ~isF && isx
                if isinf(size(x,1))
                    F	=   matfun(@(y)F*y,size(F));
                    b	=   F*x;
                else
                    if numel(F) == size(x,1) % intepret x as a vector
                        if ~isempty(x.uops.ctranspose)
                            b	=   (x'*F')'; % F*x = (x'*F')'
                        else
                            error( ['Left hand side multiplication by a vector '...
                                    'is undefined without ctranspose.'] );
                        end
                    elseif size(F,2) == size(x,1)
                        F	=   matfun(@(y)F*y,size(F)); % F is a matrix
                        b	=   F*x; % Compose matrix mult. of F and x
                    else
                        err(fliplr(size(x)),[numel(F),1]);
                    end
                end
            end
            
        end
        
        %==================================================================
        % setop
        %==================================================================
        function F = setop(F,varargin)
            
            if isempty(varargin)
                return
            end
            
            if length(varargin) == 1 && isa(varargin{1},'struct')
                [u,v]	=	deal( fieldnames(varargin{1}), struct2cell(varargin{1}) );
                ops     =   reshape( [u,v].', [], 1 );
                F       =   set_op(F,ops{:});
                return
            end
            
            if mod(length(varargin),2)
                error( 'Settings must be given as flag/value pairs.' );
            end
            
            % User settings
            boplist	=   fields(F.bops);
            uoplist	=   fields(F.uops);
            goplist =	fields(F.gops);
            for ii = 1:2:length(varargin)
                
                op	=	varargin{ii};
                fun	=	varargin{ii+1};
                ib	=	find(strcmpi(op,boplist),1);
                iu	=	find(strcmpi(op,uoplist),1);
                ig	=	find(strcmpi(op,goplist),1);
                                
                if isempty(ib) && isempty(iu) && isempty(ig)
                    error( 'Invalid operation ''%s''.', op );
                elseif ~isa(fun,'function_handle')
                    error( 'Operation must be a function handle.' );
                else
                    if isempty(iu) && isempty(ig)
                        F.bops.(boplist{ib}) = binary_matfun_op(F,fun);
                    elseif isempty(ib) && isempty(ig)
                        F.uops.(uoplist{iu}) = unary_matfun_op(F,fun);
                    else
                        F.gops.(goplist{ig}) = general_op(F,fun);
                    end
                end
                
            end
            
        end
        
        %==================================================================
        % set_default_ops
        %==================================================================
        function F = set_default_ops(F)
            F	=   setop( F,   ...
                'minus',    @default_minus,     ...
                'mpower',	@default_mpower,	...
                'plus',     @default_plus,      ...
                'shift',    @default_shift,     ...
                'times',    @default_times,     ...
                'conj',     @default_conj,      ...
                'real',     @default_real,      ...
                'imag',     @default_imag,      ...
                'uminus',	@default_uminus,	...
                'uplus',	@default_uplus      ...
                );
        end
        
        %==================================================================
        % size
        %==================================================================
        function Asize = size(F,dim)
            if nargin < 2
                Asize	=   [F.m,F.n];
            elseif ( dim == 1 )
                Asize	=   F.m;
            elseif ( dim == 2 )
                Asize	=   F.n;
            elseif ( dim > 2 && dim == round(dim) )
                Asize	=   1;
            else
                error( 'Invalid dimension: %0.2f', dim );
            end
        end
                
        %==================================================================
        % length
        %==================================================================
        function Asize = length(F)
            Asize	=   max(F.m,F.n);
        end
        
        %==================================================================
        % display
        %==================================================================
        function disp(F)
            fprintf('  %dx%d ',size(F,1),size(F,2));
            cprintf('-keyword','matfun ');
            fprintf('array with properties:\n\n');
            fprintf('\tf:  %s',func2str(F.f));
            fprintf('\n\n');
        end
        
        %==================================================================
        % isequal
        %==================================================================
        function bool = isequal(F,G)
            bool	=	isequal( F.f, G.f )	&&	...
                        isequal( F.m, G.m )	&&	...
                        isequal( F.n, G.n ) &&  ...
                        isequal( F.bops, G.bops );
        end
        
        %==================================================================
        % overloaded functions (binary operations)
        %==================================================================
        
        %------------------------------------------------------------------
        % minus
        %------------------------------------------------------------------
        function F = minus(F,G,varargin)
            if ismatfun(F)
                F	=	F.bops.minus(F,G,varargin{:});
            else
                F	=	G.bops.minus(F,G,varargin{:});
            end
        end
        
        %------------------------------------------------------------------
        % mpower
        %------------------------------------------------------------------
        function F = mpower(F,G,varargin)
            if ismatfun(F)
                F	=	F.bops.mpower(F,G,varargin{:});
            else
                F	=	G.bops.mpower(F,G,varargin{:});
            end
        end
        
        %------------------------------------------------------------------
        % plus
        %------------------------------------------------------------------
        function F = plus(F,G,varargin)
            if ismatfun(F)
                F	=	F.bops.plus(F,G,varargin{:});
            else
                F	=	G.bops.plus(F,G,varargin{:});
            end
        end
        
        %------------------------------------------------------------------
        % shift
        %------------------------------------------------------------------
        function F = shift(F,G,varargin)
            if ismatfun(F)
                F	=	F.bops.shift(F,G,varargin{:});
            else
                F	=	G.bops.shift(F,G,varargin{:});
            end
        end
        
        %------------------------------------------------------------------
        % times
        %------------------------------------------------------------------
        function F = times(F,G,varargin)
            if ismatfun(F)
                F	=	F.bops.times(F,G,varargin{:});
            else
                F	=	G.bops.times(F,G,varargin{:});
            end
        end
        
        %==================================================================
        % overloaded functions (unary operations)
        %==================================================================
        
        %------------------------------------------------------------------
        % abs
        %------------------------------------------------------------------
        function F = abs(F,varargin)
            F	=	F.uops.abs(F,varargin{:});
        end
                
        %------------------------------------------------------------------
        % conj
        %------------------------------------------------------------------
        function F = conj(F,varargin)
            F	=	F.uops.conj(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % ctranspose
        %------------------------------------------------------------------
        function F = ctranspose(F,varargin)
            F	=	F.uops.ctranspose(F,varargin{:});
            F	=   setsize(F,fliplr(size(F)));
        end
        
        %------------------------------------------------------------------
        % imag
        %------------------------------------------------------------------
        function F = imag(F,varargin)
            F	=	F.uops.imag(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % inv
        %------------------------------------------------------------------
        function F = inv(F,varargin)
            F	=	F.uops.inv(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % real
        %------------------------------------------------------------------
        function F = real(F,varargin)
            F	=	F.uops.real(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % reshape
        %------------------------------------------------------------------
        function F = reshape(F,varargin)
            F	=	F.uops.reshape(F,varargin{:});
            F	=   setsize(F,varargin{:});
        end
                
        %------------------------------------------------------------------
        % transpose
        %------------------------------------------------------------------
        function F = transpose(F,varargin)
            F	=	F.uops.transpose(F,varargin{:});
            F	=   setsize(F,fliplr(size(F)));
        end
        
        %------------------------------------------------------------------
        % uminus
        %------------------------------------------------------------------
        function F = uminus(F,varargin)
            F	=	F.uops.uminus(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % uplus
        %------------------------------------------------------------------
        function F = uplus(F,varargin)
            F	=	F.uops.uplus(F,varargin{:});
        end
        
        %==================================================================
        % overloaded functions (general operations)
        %==================================================================
                
        %------------------------------------------------------------------
        % class
        %------------------------------------------------------------------
        function F = class(F,varargin)
            F	=	F.gops.class(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % diag
        %------------------------------------------------------------------
        function F = diag(F,varargin)
            F	=	F.gops.diag(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % eps
        %------------------------------------------------------------------
        function F = eps(F,varargin)
            F	=	F.gops.eps(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % isbanded
        %------------------------------------------------------------------
        function F = isbanded(F,varargin)
            F	=	F.gops.isbanded(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % isdiag
        %------------------------------------------------------------------
        function F = isdiag(F,varargin)
            F	=	F.gops.isdiag(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % isfinite
        %------------------------------------------------------------------
        function F = isfinite(F,varargin)
            F	=	F.gops.isfinite(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % ishermitian
        %------------------------------------------------------------------
        function F = ishermitian(F,varargin)
            F	=	F.gops.ishermitian(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % isinf
        %------------------------------------------------------------------
        function F = isinf(F,varargin)
            F	=	F.gops.isinf(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % isnan
        %------------------------------------------------------------------
        function F = isnan(F,varargin)
            F	=	F.gops.isnan(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % issparse
        %------------------------------------------------------------------
        function F = issparse(F,varargin)
            F	=	F.gops.issparse(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % issymmetric
        %------------------------------------------------------------------
        function F = issymmetric(F,varargin)
            F	=	F.gops.issymmetric(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % istril
        %------------------------------------------------------------------
        function F = istril(F,varargin)
            F	=	F.gops.istril(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % istriu
        %------------------------------------------------------------------
        function F = istriu(F,varargin)
            F	=	F.gops.istriu(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % mldivide
        %------------------------------------------------------------------
        function F = mldivide(F,x,varargin)
            F	=	F.gops.mldivide(F,x,varargin{:});
        end
        
        %------------------------------------------------------------------
        % mrdivide
        %------------------------------------------------------------------
        function F = mrdivide(y,F,varargin)
            F	=	F.gops.mrdivide(y,F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % norm
        %------------------------------------------------------------------
        function F = norm(F,varargin)
            F	=	F.gops.norm(F,varargin{:});
        end
        
        %------------------------------------------------------------------
        % trace
        %------------------------------------------------------------------
        function F = trace(F,varargin)
            F	=	F.gops.trace(F,varargin{:});
        end
        
    end
        
    methods (Access = private)
        
        %==================================================================
        % setsize: Set F size
        %==================================================================
        function F = setsize(F,varargin)
            if length(varargin) == 1
                if numel(varargin{1}) ~= 2, error('F must be 2D.'); end
                [mm,nn]	=   deal(varargin{1}(1),varargin{1}(2));
            elseif length(varargin) == 2
                [mm,nn]	=   deal(varargin{:});
                if ~(numel(mm)==1 && numel(nn)==1), error('F must be 2D.'); end
            else
                error('F must be 2D.');
            end
            F.m	=   mm;
            F.n	=   nn;                
        end
        
        %==================================================================
        % default overloaded functions
        %==================================================================
        
        %------------------------------------------------------------------
        % Binary ops
        %------------------------------------------------------------------
        % (A-B)*x = A*x-B*x
        function b = default_minus(F,G,x)
            b	=   F*x-G*x;
        end
        
        % (A^n)*x = A*(A*(...*(A*x)...))
        function b = default_mpower(F,n,x)
            if isa(n,'matfun'), error('mpower A^n undefined for matfun G.'); end
            if ~isscalar(n) && (n>=0) && (n==round(n))
                error('exponent n must be a positive integer.');
            end
            b	=	expBySquaring(F,n,1)*x;
        end
        
        % (A+B)*x = A*x+B*x
        function b = default_plus(F,G,x)
            b	=   F*x+G*x;
        end
        
        % (A-mu*I)*x = A*x-mu*x
        function b = default_shift(F,mu,x)
            b	=   F*x-mu*x;
        end
        
        % (A.*s)*x = (s.*A)*x = s.*(A*x) for scalar s
        function b = default_times(F,s,x)
            if isa(s,'matfun')
                b = default_times(s,F,x);
                return
            end
            if ~(isscalar(s) && isfloat(s)), error('.* only defined for scalar s.'); end
            b	=   F*x;
            b	=   s.*b;
        end
        
        %------------------------------------------------------------------
        % Unary ops
        %------------------------------------------------------------------
        % conj(A)*x = conj(A*conj(x))
        function b = default_conj(F,x)
            b	=   conj(F*conj(x));
        end
        
        % imag(A)*x = -0.5i*(A*x-conj(A)*x)
        function b = default_imag(F,x)
            b	=   F*x;
            b	=  -0.5i*(b-conj(F*conj(x)));
        end
        
        % real(A)*x = 0.5*(A*x+conj(A)*x)
        function b = default_real(F,x)
            b	=   F*x;
            b	=   0.5*(b+conj(F*conj(x)));
        end
        
        % (-A)*x = -(A*x)
        function b = default_uminus(F,x)
            b	=  -(F*x);
        end
        
        % (+A)*x = A*x
        function b = default_uplus(F,x)
            b	=   F*x;
        end
        
        %==================================================================
        % get handle for creating matfun from binary op on pair of matfuns
        %==================================================================
        function F = binary_matfun_op(F,fun)
            F	=	@(FF,GG,varargin) matfun( @(x)fun(FF,GG,x,varargin{:}), size(F) );    
        end
        
        %==================================================================
        % get handle for creating matfun from unary op on a single matfun
        %==================================================================
        function F = unary_matfun_op(F,fun)
            F	=	@(FF,varargin) matfun( @(x)fun(FF,x,varargin{:}), size(F) );    
        end
        
        %==================================================================
        % get handle for evaluating general operator on a matfun
        %==================================================================
        function F = general_op(~,fun)
            F	=	@(FF,varargin) fun(FF,varargin{:});    
        end
        
    end
    
    methods (Static)
        
        %==================================================================
        % eye
        %==================================================================
        function I = eye(varargin)
            if nargin == 1
                % I = eye(Isize)
                Isize	=   varargin{1};
            elseif nargin == 2
                % I = eye(m,n)
                Isize	=   [varargin{1},varargin{2}];
            elseif nargin == 3 && strcmpi(varargin{2},'like')
                % I = eye(Isize,'like',F)
                Isize	=   varargin{1};
            else
                error('Invalid arguments to eye.');
            end
            I	=   matfun( @(x) x, Isize );
        end
        
        %==================================================================
        % default binary operations
        %==================================================================
        function bops = default_bops
            bops	=	struct(     ...
                'minus',        [],	...
                'mpower',       [],	...
                'plus',         [],	...
                'shift',        [],	...
                'times',        []  ...
                );
        end
        
        %==================================================================
        % default unary operations
        %==================================================================
        function uops = default_uops
            uops	=	struct(     ...
                'abs',          [],	...
                'conj',         [],	...
                'ctranspose',	[],	...
                'imag',         [],	...
                'inv',          [],	...
                'real',         [],	...
                'reshape',      [],	...
                'transpose',	[],	...
                'uminus',       [],	...
                'uplus',        []	...
                );
        end
        
        %==================================================================
        % default general operations
        %==================================================================
        function gops = default_gops
            gops	=	struct(     ...
                'class',        [], ...
                'diag',         [], ...
                'eps',          [],	...
                'isbanded',     [],	...
                'isdiag',       [],	...
                'isfinite',     [],	...
                'ishermitian',	[],	...
                'isinf',        [],	...
                'isnan',        [],	...
                'issparse',     [],	...
                'issymmetric',	[],	...
                'istril',       [],	...
                'istriu',       [],	...
                'mldivide',     [],	...
                'mrdivide',     [],	...
                'norm',         [],	...
                'trace',        []  ...
                );
        end
                
    end
    
end
