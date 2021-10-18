function [x] = fmg_relax_gs_diffuse(x, b, maxIter, h, D, f)
    
    if nargin < 6 || isempty(f)
        f = zeros(size(x),'double');
    elseif isscalar(f)
        f = double(f)*ones(size(x),'double');
    elseif isequal(size(f),size(x))
        if ~isa(f,'double'), f = double(f); end
    else
        error('f must be a scalar or have the same size as x.');
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
    
    if nargin < 3, maxIter = 20; end
    
    if isscalar(b)
        b = double(b)*ones(size(x),'double');
    elseif isequal(size(b),size(x))
        if ~isa(b,'double'), b = double(b); end
    else
        error('b must be a scalar or have the same size as x.');
    end
    
    %----------------------------------------------------------------------
    
    isSingle	=   isa(x,'single');
    isDouble	=   isa(x,'double');
    isReal      =   isreal(x);
    
    if ~(isSingle || isDouble), error('x must be double or single.'); end
    if isSingle, x = double(x); end
    if isReal, x = complex(x); end
    
    d = D/h^2;
    g = 1./(f+6*d);
    c = b.*g;
    g = d.*g;
    
    if isreal(c), c = complex(c); end
    if isreal(g), g = complex(g); end
    
    x = fmg_relax_gs_diffuse_cd(x, c, maxIter, g);
    
    if isSingle, x = single(x); end
    

end

