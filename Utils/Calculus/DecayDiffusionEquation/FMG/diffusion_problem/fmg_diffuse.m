function y = fmg_diffuse(x, h, D, f, c)
%FMG_DIFFUSE Discrete decay diffusion equation operator with 2nd order
% central difference approximations on derivatives and periodic boundary
% conditions. The operator is:
% 
%   L[x] = div(D*grad(x)) - f*x    % for a tensor D
%        = D*lap(x) - f*x          % for scalar D
% 
% Currently, D must be a scalar and f must be a 3D array of the same size
% as x. The result may be multiplied by an arbitrary real scalar constant c
% as well, if provided.
    
    if nargin < 5 || isempty(c)
        c = 1;
    elseif ~isscalar(c) || ~isreal(c)
        error('c must be a constant real scalar.');
    end
    
    if nargin < 4 || isempty(f)
        f = zeros(size(x),'double');
    elseif isscalar(f)
        f = double(f)*ones(size(x),'double');
        if isreal(f), f = complex(f); end
    elseif isequal(size(f),size(x))
        if ~isa(f,'double'), f = double(f); end
        if isreal(f), f = complex(f); end
    else
        error('f must be a scalar or have the same size as x.');
    end
    
    if nargin < 3 || isempty(D)
        D = 1;
    elseif ~(isscalar(D) && isfloat(D))
        error('D must be a floating point scalar.');
    else
        D = double(D(1));
    end
    
    if nargin < 2 || isempty(h)
        h = 1;
    else
        h = double(h(1));
    end
    
    dim	= ndims(x);
    if ~( dim == 3 || dim == 4 )
        error( 'Dimension of x must be 3 or 4' );
    end
    
    %----------------------------------------------------------------------
    
    isSingle	=   isa(x,'single');
    isDouble	=   isa(x,'double');
    isReal      =   isreal(x);
    
    if ~(isSingle || isDouble), error('x must be double or single.'); end
    if isSingle, x = double(x); end
    if isReal, x = complex(x); end
    
    y = fmg_diffuse_cd(x, h, D, f, c);
    
    if isSingle, y = single(y); end
    
end
