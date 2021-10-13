function y = sor_diffuse(b, x, w, h, D, f, s, c, iter, dir)
%SOR_DIFFUSE Successive-overrelaxaiton scheme for the Bloch-Torrey
% operator with 2nd order central difference spatial derivatives and
% periodic boundary conditions. The operator is:
% 
%   L[x] = div(D*grad(x)) - f*x    % for a tensor D
%        = D*lap(x) - f*x          % for scalar D
% 
% And so the solution y is an approximate solution to:
% 
%   L[y] = b
% 
% After 'iter' iterations of SOR smoothing with parameter 'w'.
% 
% Currently, D must be a scalar and f must be a 3D array of the same size
% as b. The result may be multiplied by an arbitrary real scalar constant c
% as well, if provided.

    if nargin < 10 || isempty(dir)
        dir = 0;
    elseif dir ~= 0
        dir = 1;
    end
    
    if nargin < 9 || isempty(iter)
        iter = 10;
    end
    
    if nargin < 8 || isempty(c)
        c = 1;
    elseif ~isscalar(c) || ~isreal(c)
        error('c must be a constant real scalar.');
    end
    
    if nargin < 7 || isempty(s)
        s = 0;
    elseif ~isscalar(s) || ~isreal(s)
        error('s must be a constant real scalar.');
    end
    
    if nargin < 6 || isempty(f)
        f = zeros(size(b),'double');
    elseif isscalar(f)
        f = double(f)*ones(size(b),'double');
        if isreal(f), f = complex(f); end
    elseif ~isequal(size(f),size(b))
        error('f must be a scalar or have the same size as b.');
    end
    
    if nargin < 5 || isempty(D)
        D = 1;
    elseif ~(isscalar(D) && isfloat(D))
        error('D must be a floating point scalar.');
    else
        D = double(D(1));
    end
    
    if nargin < 4 || isempty(h)
        h = 1;
    else
        h = double(h(1));
    end
    
    if nargin < 3 || isempty(w)
        w = 1;
    end
    
    if nargin < 2 || isempty(x)
        x = complex(zeros(size(b)));
    elseif ~isequal(size(x),size(b))
        error('x must have the same size as b.');
    end
    
    if ndims(b) ~= 3
        error( 'Dimension of b must be 3.' );
    end
    
    if isa(b,'single'); b = double(b); end
    if isa(x,'single'); x = double(x); end
    if isa(f,'single'); f = double(f); end
    
    if isreal(b); b = complex(b); end
    if isreal(x); x = complex(x); end
    if isreal(f); f = complex(f); end
        
    %----------------------------------------------------------------------
    
    y = sor_diffuse_cd(b, x, w, h, D, f, s, c, iter, dir);
    
end

