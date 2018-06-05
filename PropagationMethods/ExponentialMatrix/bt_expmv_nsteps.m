function [ x, signal, time ] = bt_expmv_nsteps( dt, A, x0, n, varargin )
%BT_EXPMV_NSTEPS Calls expmv for the BlochTorreyOp for n small timesteps dt

p = getInputParser;
parse(p,varargin{:})

opts = p.Results;
selectdegargs = {opts.prec,opts.shift,opts.bal,opts.force_estm,opts.force_no_estm};
expmvargs = {opts.prec,opts.shift,opts.bal,opts.full_term,opts.prnt};

time = dt*(0:n).';
scalesum = prod(A.gdims)/prod(A.gsize);
signal = [];

if strcmpi(opts.calcsignal,'all')
    signal = zeros(n+1,1);
    signal(1) = scalesum * sum(x0);
end

x = x0(:);
M = select_taylor_degree(A,x,[],[],selectdegargs{:});

for jj = 1:n
    looptime = tic;
    
    x = expmv(dt,A,x,M,expmvargs{:});
    
    if strcmpi(opts.calcsignal,'all')
        signal(jj+1) = scalesum * sum(x);
    end
    
    if ( 2*jj == n ) && strcmpi( opts.type, 'SE' )
        x = conj(x);
    end
    
    str = sprintf('t = %4.1fms', 1000*dt*jj);
    if opts.prnt, display_toc_time(toc(looptime),str); end
end

if strcmpi(opts.calcsignal,'last')
    signal = scalesum * sum(x);
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
calcsignalopts = {'all','last','none'};
addParameter(p,'calcsignal','all',@(x) any(VS(x,calcsignalopts)));
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

end

