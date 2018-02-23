function [ x, sig ] = bt_expmv_tspan( tspan, A, x0, type, scale )
%BT_EXPMV_TSPAN expmv for linearly spaced values tspan for BlochTorreyOp A.
% NOTE: tspan must be in seconds.

tspan = tspan(:).';
if ~( tspan(1)==0 && all(diff(tspan,2)<5*eps(tspan(end))) && all(diff(tspan)>0) )
    error('tspan must be linearly spaced and in increasing order, starting from zero.');
end

n = numel(tspan)-1;
dt = mean(diff(tspan));

if strcmpi(type,'SE') && ~(n == 2*round(n/2))
    error('must be even number of time steps for SE simulations');
end

prec = 'double';
shift = true;
bal = false;
force_estm = false;
full_term = false;
prnt = true;
if nargin < 5; scale = prod(A.h); end

x = x0(:);
sig = complex(zeros(size(tspan)));
sig(1) = scale * sum(x);

degree_time = tic;
M = select_taylor_degree(dt*A,x,[],[],prec,shift,bal,force_estm);
display_toc_time( toc(degree_time), 'Selecting Taylor Degree' );

start_time = tic;
for ii = 2:n+1
    
    %     t = tspan(ii);
    %
    %     degree_time = tic;
    %     M = select_taylor_degree(t*A,x,[],[],prec,shift,bal,force_estm);
    %     display_toc_time( toc(degree_time), 'Selecting Taylor Degree' );
    %
    %     expmv_time = tic;
    %     x = expmv(t,A,x,M,prec,shift,bal,full_term,prnt);
    %     display_toc_time( toc(expmv_time), 'Expmv Iteration' );
    
    expmv_time = tic;
    x = expmv(dt,A,x,M,prec,shift,bal,full_term,prnt);
    display_toc_time( toc(expmv_time), 'Expmv Iteration' );
    
    if strcmpi(type,'SE') && (2*(ii-1) == n)
        x = conj(x);
        fprintf('Conjugating.\n');
    end
    sig(ii) = scale * sum(x);
    
    display_toc_time( toc(start_time), ...
        sprintf( 'step #%d/%d, t = %2.2fms, S = %1.4e + %1.4ei', ...
        ii-1, n, 1000 * tspan(ii), real(sig(ii)), imag(sig(ii)) ), ...
        [0,1] );
end

x = reshape(x,size(x0));

end

