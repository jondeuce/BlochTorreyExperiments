function [ options ] = parabolicATSoptions( x0, T, varargin )
%PARABOLICATSOPTIONS Get options structure for parabolic ATS routine

if nargin < 1 || isempty(x0), x0 = 1; end
if nargin < 2 || isempty(T),  T  = 1; end

if length(varargin) == 1 && isa(varargin{1},'struct')
    [f,v]	=	deal( fieldnames(varargin{1}), struct2cell(varargin{1}) );
    opts	=   reshape( [f,v].', [], 1 );
    options	=   parabolicATSoptions(x0,T,opts{:});
    return
end

if mod(length(varargin),2)
    error( 'Settings must be given as flag/value pairs.' );
end

% Load default options
options     =	struct(	...
    'StepScheme',   'Richardson',       ...
    'AbsTol',       1.0e-3 * norm(x0),  ...
    'RelTol',       1.0e-3,             ...
    'InitialStep',  1.0e-2 * T,         ...
    'MaxStep',      1.0e-1 * T,         ...
    'MinStep',      1.0e-3 * T,         ...
    'SubSteps',     4,                  ...
    'Verbose',      false               ...
    );

% User settings
optfields	=   fields(options);
for ii = 1:2:length(varargin)
    
    flag	=   varargin{ii};
    value	=   varargin{ii+1};
    idx     =   find(strcmpi(flag,optfields),1);
    
    if isempty(idx)
        warning( 'Unknown option ''%s''. Using default.', flag );
    else
        options.(optfields{idx})	=   value;
    end
    
end

end

