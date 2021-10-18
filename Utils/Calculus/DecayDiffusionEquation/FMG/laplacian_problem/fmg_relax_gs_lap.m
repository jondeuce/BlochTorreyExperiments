function [x] = fmg_relax_gs_lap(x, b, maxIter, h)
    
    if nargin < 5 || isempty(h)
        h = 1;
    end
    if isscalar(h)
        h = [h, h, h];
    end
    if nargin < 4, maxIter = 500; end
    
    Mask = true(size(x));
    if isa(x, 'single')
        if ~isa(b, 'single'), b = single(b); end
        x = gs_lap_mex_s(x, b, Mask, maxIter, h);
    elseif isa(x, 'double')
        if ~isa(b, 'double'), b = double(b); end
        x = gs_lap_mex_d(x, b, Mask, maxIter, h);
    else
        error('x must be double or single');
    end

end

