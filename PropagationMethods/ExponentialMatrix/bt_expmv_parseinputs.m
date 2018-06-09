function [opts, selectdegargs, expmvargs] = bt_expmv_parseinputs(varargin)
%BT_EXPMV_PARSEINPUTS [opts, selectdegargs, expmvargs] = bt_expmv_parseinputs(varargin)

p = bt_expmv_inputparser;
parse(p,varargin{:})

opts = p.Results;
selectdegargs = {opts.m_max,opts.p_max,opts.prec,opts.shift,opts.bal,opts.force_estm,opts.force_no_estm};
expmvargs = {opts.M,opts.prec,opts.shift,opts.bal,opts.full_term,opts.prnt,opts.m_min};

end

function p = bt_expmv_inputparser

p = inputParser;
p.FunctionName = mfilename;

VA = @(varargin) validateattributes(varargin{:});
VS = @(varargin) validatestring(varargin{:});

% BlochTorrey args
addParameter(p,'type','gre',@(x)VA(x,{'char'},{'nonempty'}));

% expmv args
checknumericprec = @(x) (isnumeric(x) && isscalar(x) && x > 0 && x < 1);
prectypes = {'double','single','half'};

addParameter(p,'prec','double',@(x)checknumericprec(x) || any(VS(x,prectypes)));
addParameter(p,'shift',true,@islogical);
addParameter(p,'bal',false,@islogical);
addParameter(p,'force_estm',false,@islogical);
addParameter(p,'force_no_estm',true,@islogical);
addParameter(p,'full_term',false,@islogical);
addParameter(p,'prnt',true,@islogical);
addParameter(p,'forcesparse',false,@islogical);
addParameter(p,'M',[],@isnumeric);
addParameter(p,'m_max',[],@isnumeric);
addParameter(p,'p_max',[],@isnumeric);
addParameter(p,'m_min',[],@isnumeric);
addParameter(p,'adapttaylor',false,@islogical);

end
