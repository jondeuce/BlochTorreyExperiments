function [x] = fmg_mu_cycle(x, b, iters, h, d, fun, relax)
	
    % Relax for iters.pre iterations
    x = relax(x,b,iters.pre,h);
    
    if d > 1
        
        % Get residual
        r = fun(x,h);
        r = b - r;
        
        r = fmg_restrict(r);
        v = zeros(size(r),'like',r);
        for jj = 1:iters.mu
            v  = fmg_mu_cycle(v, r, iters, 2*h, d-1, fun, relax);
        end
        x = x + fmg_prolong(v);
        
    end
    
    % Relax for iters.post iterations
    x = relax(x,b,iters.post,h);
    
end
