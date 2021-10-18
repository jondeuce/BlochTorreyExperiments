function x = FMG(b, iters, d, fun, relax )
% FMG Full-multigrid solver for the problem defined by fun(x,h) = b, where
% x is the solution variable and h is the gridsize (1 at finest grid).
% relax(x,b,ii,h) relaxes the solution x on gridsize h for ii iterations.
% Both fun(x,h) and relax(x,b,ii,h) must handle both cases for single and
% double.
	
    if nargin < 5 || ~(isa(relax,'function_handle') && nargin(relax)==4)
        error('relax = @(x,b,ii,h)r(x,b,ii,h) must be a function handle of 4 vars.');
    end
    
    if nargin < 4 || ~(isa(fun,'function_handle') && nargin(fun)==2)
        error('fun = @(x,h)f(x,h) must be a function handle of 2 vars.');
    end
	
    if nargin < 3 || isempty(d)
        d = fmg_max_depth(b);
    end
    
    if nargin < 2 || isempty(iters)
        iters = [];
    end

    if ~isstruct(iters) && ~isempty(iters)
        iters = struct('pre', iters, 'post', iters);
    end
    
    if ~isfield(iters, 'mg')
        iters.mg = 1;
    end
    
    if ~isfield(iters, 'mu')
        iters.mu = 1;
    end
    
    if ~isfield(iters, 'pre')
        iters.pre = 20;
    end
    
    if ~isfield(iters, 'post')
        iters.post = 20;
    end
    
    x = FMG_(b, iters, 1, d, fun, relax);
    
end

function [x] = FMG_(b, iters, h, d, fun, relax)
    
    if d > 1
        b2 = fmg_restrict(b);
        x  = FMG_(b2, iters, 2*h, d-1, fun, relax);
        x  = fmg_prolong(x);
    else
        x  = zeros(size(b),'like',b);
    end
    
    for jj = 1:iters.mg
        x  = fmg_mu_cycle(x, b, iters, h, d, fun, relax);
    end
    
end
