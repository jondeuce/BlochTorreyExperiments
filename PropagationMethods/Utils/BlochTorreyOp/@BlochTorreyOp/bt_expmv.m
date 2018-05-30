function [ x ] = bt_expmv( t, A, x0, varargin )
%BT_EXPMV calls expmv for the BlochTorreyOp A

p = getInputParser;
parse(p,varargin{:})

opts = p.Results;
selectdegargs = {opts.prec,opts.shift,opts.bal,opts.force_estm,opts.force_no_estm};
expmvargs = {opts.prec,opts.shift,opts.bal,opts.full_term,opts.prnt};

if opts.forcesparse
    A = sparse(A);
end

x = x0(:);
switch upper(opts.type)
    case 'GRE'
        %M = select_taylor_degree(t*A,x,[],[],selectdegargs{:}); %TODO: not correct pre-computation?
        M = [];
        x = expmv(t,A,x,M,expmvargs{:});
    case 'SE'
        M = select_taylor_degree((t/2)*A,x,[],[],selectdegargs{:});
        x = expmv(t/2,A,x,M,expmvargs{:});
        x = conj(x);
        x = expmv(t/2,A,x,M,expmvargs{:});
    otherwise
        error('type must be either ''SE'' or ''GRE''.');
end

if ~isequal(size(x),size(x0))
    x = reshape(x,size(x0));
end

end


function p = getInputParser

p = inputParser;
p.FunctionName = mfilename;

VA = @(varargin) validateattributes(varargin{:});
VS = @(varargin) validatestring(varargin{:});

% BlochTorrey args
addParameter(p,'type','gre',@(x)VA(x,{'char'},{'nonempty'}));

% expmv args
prectypes = {'double','single','half'};
addParameter(p,'prec','double',@(x)any(VS(x,prectypes)));
addParameter(p,'shift',true,@islogical);
addParameter(p,'bal',false,@islogical);
addParameter(p,'force_estm',false,@islogical);
addParameter(p,'force_no_estm',true,@islogical);
addParameter(p,'full_term',false,@islogical);
addParameter(p,'prnt',true,@islogical);
addParameter(p,'forcesparse',false,@islogical);

end

