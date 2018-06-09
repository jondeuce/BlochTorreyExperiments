function [ x, s, m, m_min ] = bt_expmv( t, A, x0, varargin )
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
        [x,s,m,~,~,~,m_min] = expmv(t,A,x0,expmvargs{:});
    case 'SE'
        [x,~,~,~,~,~,m_min1] = expmv(t/2,A,x0,expmvargs{:});
        x = conj(x);
        [x,s,m,~,~,~,m_min2] = expmv(t/2,A,x,expmvargs{:});
        m_min = min(m_min1, m_min2);
    otherwise
        error('type must be either ''SE'' or ''GRE''.');
end

end


