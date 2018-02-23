function [T, S, M] = propconvdiff( tspan, A, x0, tol, type )
%PROPCONVDIFF

starttime = tic;

% [T_last, S_last, M_last] = propconvdiff_nsteps( A, 1, diff(tspan), x0, type );
[~, S_last, ~] = propconvdiff_nsteps( A, 1, diff(tspan), x0, type );
[T, S, M] = propconvdiff_nsteps( A, 2, diff(tspan)/2, x0, type );

dt = diff(tspan)/2;
n = 2;
iter = 0;

err = abs(S(end)-S_last(end));
maxS = max(abs(S(end)),abs(S_last(end)));
str = sprintf('iter = %d, rel-err = %1.6e', iter, err/maxS);
display_toc_time(toc(starttime),str,[0,1]);

while err >= tol*maxS
    iter = iter + 1;
    dt = dt/2;
    n = 2*n;
    S_last = S;
    %T_last = T;
    %M_last = M;
    
    [T, S, M] = propconvdiff_nsteps( A, n, dt, x0, type );
    
    err = abs(S(end)-S_last(end));
    maxS = max(abs(S(end)),abs(S_last(end)));
    str = sprintf('iter = %d, rel-err = %1.6e', iter, err/maxS);
    display_toc_time(toc(starttime),str,[0,1]);
end

end

