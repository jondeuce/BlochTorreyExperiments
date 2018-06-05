function [ x ] = bt_expmv( t, A, x0, varargin )
%BT_EXPMV calls expmv using the matrix or object A, forming the exponential
%matrix-vector product expm(A*t)*x0 without explicity calculating the full
%matrix A. Uses Higham et al.'s expmv program.

if numel(varargin) == 3
    % Inputs are parsed opts struct and selectdegargs/expmvargs cell arrays
    [opts, selectdegargs, expmvargs] = deal(varargin{:});
else
    % Inputs are name-value pairs; parse and return
    [opts, selectdegargs, expmvargs] = bt_expmv_parseinputs(varargin{:});
end

if isempty(opts.M)
    expmvargs{1} = select_taylor_degree(A,x0,selectdegargs{:});
end

if opts.forcesparse
    A = sparse(A);
end

switch upper(opts.type)
    case 'GRE'
        x = expmv(t,A,x0,expmvargs{:});
    case 'SE'
        x = expmv(t/2,A,x0,expmvargs{:});
        x = conj(x);
        x = expmv(t/2,A,x,expmvargs{:});
    otherwise
        error('type must be either ''SE'' or ''GRE''.');
end

end


