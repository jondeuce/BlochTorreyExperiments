classdef WrappedMatrix
    %WRAPPEDMATRIX Simply wraps the matrix A into a class. Mainly for
    %testing cost of overhead of class wrappers.
    
    properties ( Access = private )
        A % Matrix
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CLASS CONSTRUCTOR:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        function f = WrappedMatrix(varargin)
            % WRAPPEDMATRIX class constructor.
            if ( (nargin == 0) || isempty(varargin{1}) )
                f.A = [];
                return
            else
                f.A = varargin{1};
            end
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% PUBLIC METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Custom methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [ out ] = sumall( f ) %SUMALL
            out = sum( f.A(:) );
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Custom overloads
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ out ] = size( f, varargin ) %SIZE
            out = size(f.A, varargin{:});
        end
                
        function [ ] = disp( f ) %DISP
            fprintf('  %dx%d WrappedMatrix array\n\n', size(f));
            fprintf('    A: [%dx%d %s]\n\n', size(f), class(f.A));
        end
        
        function [ A ] = full( f ) %FULL
            A = f.A;
        end
        
        function [ out ] = length( f ) %LENGTH
            out = length(f.A);
        end
        
        function [ out ] = diag( f, varargin ) %DIAG
            out = diag(f.A, varargin{:});
        end
        
        function [ out ] = norm( f, varargin ) %NORM
            out = norm(f.A, varargin{:});
        end
                
        function [ out ] = sum( f, varargin ) %SUM
            out = sum(f.A, varargin{:});
        end
        
        function [ out ] = max( f, varargin ) %MAX
            out = max(f.A, varargin{:});
        end
        
        function [ out ] = min( f, varargin ) %MIN
            out = min(f.A, varargin{:});
        end
                
        function [ f ] = fft( f, varargin ) %FFT
            f.A = fft(f.A, varargin{:});
        end
        
        function [ f ] = fftn( f, varargin ) %FFTN
            f.A = fftn(f.A, varargin{:});
        end
        
        function [ bool ] = isscalar( f )
            bool = isscalar(f.A);
        end
        
        function [ bool ] = isnumeric( f )
            bool = isnumeric(f.A);
        end
        
        function [ bool ] = isfloat( f )
            bool = isfloat(f.A);
        end
        
        function [ bool ] = isempty( f )
            bool = isempty(f.A);
        end
        
        function [ bool ] = isreal( f )
            bool = isreal(f.A);
        end
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Auto-generated: unary operators 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ f ] = abs( f ) %ABS
            f.A = abs( f.A );
        end

        function [ f ] = acos( f ) %ACOS
            f.A = acos( f.A );
        end

        function [ f ] = acosd( f ) %ACOSD
            f.A = acosd( f.A );
        end

        function [ f ] = acosh( f ) %ACOSH
            f.A = acosh( f.A );
        end

        function [ f ] = acot( f ) %ACOT
            f.A = acot( f.A );
        end

        function [ f ] = acotd( f ) %ACOTD
            f.A = acotd( f.A );
        end

        function [ f ] = acoth( f ) %ACOTH
            f.A = acoth( f.A );
        end

        function [ f ] = acsc( f ) %ACSC
            f.A = acsc( f.A );
        end

        function [ f ] = acscd( f ) %ACSCD
            f.A = acscd( f.A );
        end

        function [ f ] = acsch( f ) %ACSCH
            f.A = acsch( f.A );
        end

        function [ f ] = asec( f ) %ASEC
            f.A = asec( f.A );
        end

        function [ f ] = asecd( f ) %ASECD
            f.A = asecd( f.A );
        end

        function [ f ] = asech( f ) %ASECH
            f.A = asech( f.A );
        end

        function [ f ] = asin( f ) %ASIN
            f.A = asin( f.A );
        end

        function [ f ] = asind( f ) %ASIND
            f.A = asind( f.A );
        end

        function [ f ] = asinh( f ) %ASINH
            f.A = asinh( f.A );
        end

        function [ f ] = atan( f ) %ATAN
            f.A = atan( f.A );
        end

        function [ f ] = atand( f ) %ATAND
            f.A = atand( f.A );
        end

        function [ f ] = atanh( f ) %ATANH
            f.A = atanh( f.A );
        end

        function [ f ] = ceil( f ) %CEIL
            f.A = ceil( f.A );
        end

        function [ f ] = conj( f ) %CONJ
            f.A = conj( f.A );
        end

        function [ f ] = cos( f ) %COS
            f.A = cos( f.A );
        end

        function [ f ] = cosd( f ) %COSD
            f.A = cosd( f.A );
        end

        function [ f ] = cosh( f ) %COSH
            f.A = cosh( f.A );
        end

        function [ f ] = cot( f ) %COT
            f.A = cot( f.A );
        end

        function [ f ] = cotd( f ) %COTD
            f.A = cotd( f.A );
        end

        function [ f ] = coth( f ) %COTH
            f.A = coth( f.A );
        end

        function [ f ] = csc( f ) %CSC
            f.A = csc( f.A );
        end

        function [ f ] = cscd( f ) %CSCD
            f.A = cscd( f.A );
        end

        function [ f ] = csch( f ) %CSCH
            f.A = csch( f.A );
        end

        function [ f ] = ctranspose( f ) %CTRANSPOSE
            f.A = ctranspose( f.A );
        end

        function [ out ] = det( f ) %DET
            out = det( f.A );
        end

        function [ out ] = dmperm( f ) %DMPERM
            out = dmperm( f.A );
        end

        function [ f ] = exp( f ) %EXP
            f.A = exp( f.A );
        end

        function [ f ] = expm1( f ) %EXPM1
            f.A = expm1( f.A );
        end

        function [ f ] = fix( f ) %FIX
            f.A = fix( f.A );
        end

        function [ f ] = floor( f ) %FLOOR
            f.A = floor( f.A );
        end

        function [ f ] = imag( f ) %IMAG
            f.A = imag( f.A );
        end

        function [ f ] = inv( f ) %INV
            f.A = inv( f.A );
        end

        function [ out ] = isdiag( f ) %ISDIAG
            out = isdiag( f.A );
        end

        function [ out ] = isfinite( f ) %ISFINITE
            out = isfinite( f.A );
        end

        function [ out ] = isinf( f ) %ISINF
            out = isinf( f.A );
        end

        function [ out ] = isnan( f ) %ISNAN
            out = isnan( f.A );
        end

        function [ out ] = istril( f ) %ISTRIL
            out = istril( f.A );
        end

        function [ out ] = istriu( f ) %ISTRIU
            out = istriu( f.A );
        end

        function [ f ] = log( f ) %LOG
            f.A = log( f.A );
        end

        function [ f ] = log10( f ) %LOG10
            f.A = log10( f.A );
        end

        function [ f ] = log1p( f ) %LOG1P
            f.A = log1p( f.A );
        end

        function [ f ] = log2( f ) %LOG2
            f.A = log2( f.A );
        end

        function [ out ] = nnz( f ) %NNZ
            out = nnz( f.A );
        end

        function [ out ] = nonzeros( f ) %NONZEROS
            out = nonzeros( f.A );
        end

        function [ out ] = nzmax( f ) %NZMAX
            out = nzmax( f.A );
        end

        function [ out ] = rcond( f ) %RCOND
            out = rcond( f.A );
        end

        function [ f ] = real( f ) %REAL
            f.A = real( f.A );
        end

        function [ f ] = sec( f ) %SEC
            f.A = sec( f.A );
        end

        function [ f ] = secd( f ) %SECD
            f.A = secd( f.A );
        end

        function [ f ] = sech( f ) %SECH
            f.A = sech( f.A );
        end

        function [ f ] = sign( f ) %SIGN
            f.A = sign( f.A );
        end

        function [ f ] = sin( f ) %SIN
            f.A = sin( f.A );
        end

        function [ f ] = sind( f ) %SIND
            f.A = sind( f.A );
        end

        function [ f ] = sinh( f ) %SINH
            f.A = sinh( f.A );
        end

        function [ f ] = sqrt( f ) %SQRT
            f.A = sqrt( f.A );
        end

        function [ out ] = symrcm( f ) %SYMRCM
            out = symrcm( f.A );
        end

        function [ f ] = tan( f ) %TAN
            f.A = tan( f.A );
        end

        function [ f ] = tand( f ) %TAND
            f.A = tand( f.A );
        end

        function [ f ] = tanh( f ) %TANH
            f.A = tanh( f.A );
        end

        function [ f ] = transpose( f ) %TRANSPOSE
            f.A = transpose( f.A );
        end

        function [ f ] = uminus( f ) %UMINUS
            f.A = uminus( f.A );
        end

        function [ f ] = uplus( f ) %UPLUS
            f.A = uplus( f.A );
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Auto-generated: binary operators 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [ out ] = amd( f, g ) %AMD
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                out = amd(f.A,g.A);
            elseif  fIsWrapped && ~gIsWrapped
                out = amd(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                out = amd(f,g.A);
            end
        end

        function [ out ] = eq( f, g ) %EQ
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                out = eq(f.A,g.A);
            elseif  fIsWrapped && ~gIsWrapped
                out = eq(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                out = eq(f,g.A);
            end
        end

        function [ out ] = ge( f, g ) %GE
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                out = ge(f.A,g.A);
            elseif  fIsWrapped && ~gIsWrapped
                out = ge(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                out = ge(f,g.A);
            end
        end

        function [ out ] = gt( f, g ) %GT
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                out = gt(f.A,g.A);
            elseif  fIsWrapped && ~gIsWrapped
                out = gt(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                out = gt(f,g.A);
            end
        end

        function [ h ] = hess( f, g ) %HESS
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(hess(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = hess(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(hess(f,g.A));
            end
        end

        function [ h ] = hypot( f, g ) %HYPOT
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(hypot(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = hypot(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(hypot(f,g.A));
            end
        end

        function [ h ] = ldivide( f, g ) %LDIVIDE
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(ldivide(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = ldivide(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(ldivide(f,g.A));
            end
        end

        function [ out ] = le( f, g ) %LE
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                out = le(f.A,g.A);
            elseif  fIsWrapped && ~gIsWrapped
                out = le(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                out = le(f,g.A);
            end
        end

        function [ out ] = lt( f, g ) %LT
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                out = lt(f.A,g.A);
            elseif  fIsWrapped && ~gIsWrapped
                out = lt(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                out = lt(f,g.A);
            end
        end

        function [ h ] = minus( f, g ) %MINUS
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(minus(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = minus(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(minus(f,g.A));
            end
        end

        function [ h ] = mldivide( f, g ) %MLDIVIDE
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(mldivide(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = mldivide(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(mldivide(f,g.A));
            end
        end

        function [ h ] = mrdivide( f, g ) %MRDIVIDE
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(mrdivide(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = mrdivide(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(mrdivide(f,g.A));
            end
        end

        function [ h ] = mtimes( f, g ) %MTIMES
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(mtimes(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = mtimes(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(mtimes(f,g.A));
            end
        end

        function [ out ] = ne( f, g ) %NE
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                out = ne(f.A,g.A);
            elseif  fIsWrapped && ~gIsWrapped
                out = ne(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                out = ne(f,g.A);
            end
        end

        function [ h ] = plus( f, g ) %PLUS
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(plus(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = plus(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(plus(f,g.A));
            end
        end

        function [ h ] = pow2( f, g ) %POW2
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(pow2(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = pow2(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(pow2(f,g.A));
            end
        end

        function [ h ] = power( f, g ) %POWER
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(power(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = power(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(power(f,g.A));
            end
        end

        function [ h ] = rdivide( f, g ) %RDIVIDE
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(rdivide(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = rdivide(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(rdivide(f,g.A));
            end
        end

        function [ h ] = times( f, g ) %TIMES
            fIsWrapped = isa(f, 'WrappedMatrix');
            gIsWrapped = isa(g, 'WrappedMatrix');
            if fIsWrapped && gIsWrapped
                h = WrappedMatrix(times(f.A,g.A));
            elseif  fIsWrapped && ~gIsWrapped
                h = times(f.A,g);
            elseif ~fIsWrapped &&  gIsWrapped
                h = WrappedMatrix(times(f,g.A));
            end
        end
        
    end
end

