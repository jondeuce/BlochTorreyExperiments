function [x] = FMG(b, Mask, iters, d, h)
%FMG [x] = FMG(b, Mask, iters, d, h)
%   Detailed explanation goes here
 
    if nargin < 2 || isempty(Mask)
        Mask = true(size(b));
    end
    
    if nargin < 4 || isempty(d)
        d = fmg_max_depth(Mask);
    end
    
    if nargin < 5
        h = 1;
    end
    
    if nargin < 3 || isempty(iters)
        iters = [];
    end
    
    if ~isstruct(iters) && ~isempty(iters)
        iters = struct('pre', iters, 'post', iters);
    end
    
    if ~isfield(iters, 'mg')
        iters.mg = 3;
    end
    
    if ~isfield(iters, 'mu')
        iters.mu = 1;
    end
    
    if ~isfield(iters, 'pre')
        iters.pre = 2;
    end
    
    if ~isfield(iters, 'post')
        iters.post = 5;
    end
    
    x = FMG_(b, Mask, iters, h, d);
    
end


function [x] = FMG_(b, G, it, h, d)
    
    if d > 1
        [b2, G2] = restrict_mex(b, G);
        x = FMG_(b2, G2, it, 2*h, d-1);
        x = prolong_mex(x, G2, G);
    else
        x = zeros(size(G), class(b));
    end
    
    for jj = 1:it.mg
        x  = fmg_mu_cycle(x, b, G, it, h, d);
    end
    
end


function [x] = fmg_mu_cycle(x, b, G, it, h, d)
    
    x = gs_forward_mex(x, b, G, it.pre, h);
    
    if d > 1
        r = b - lap(x, G, h);
        [r, G2] = restrict_mex(r, G);
        v = zeros(size(r), class(r));
        for jj = 1:it.mu
            v  = fmg_mu_cycle(v, r, G2, it, 2*h, d-1);
        end
        x = x + prolong_mex(v, G2, G);
    end
    
    x = gs_backward_mex(x, b, G, it.post, h);
    
end


function [d] = fmg_max_depth(G)
    
    mSize = size(crop2mask(G));
    d = max(min(floor(log2(mSize))) - 2, 1);
    
end
