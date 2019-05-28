function [ args ] = parseinputs(BTStepper, varargin)
%PARSEINPUTS [ Stepper ] = parseinputs(Stepper, kwargs)

if nargin < 2; varargin = {}; end

RequiredArgs = { 'TimeStep', 'Dcoeff', 'Gamma', 'dGamma', 'GridSize', 'VoxelSize' };

DefaultArgs   =   struct(...
    'Order', 2, ...
    'NReps', 1, ...
    'allowPreCompConvKernels', true, ...
    'allowPreCompExpArrays', true, ...
    'useGaussianKernels', true
    );

p = getParser(DefaultArgs, RequiredArgs);
parse(p, varargin{:});
args = p.Results;
args = appendDerivedQuantities(args);

end

function p = getParser(DefaultArgs, RequiredArgs)

p = inputParser;

for f = RequiredArgs
    paramName = f{1};
    addRequired(p,paramName)
end

for f = fieldnames(DefaultArgs).'
    paramName = f{1};
    defaultVal = DefaultArgs.(f{1});
    addParameter(p,paramName,defaultVal)
end

end

function args = appendDerivedQuantities(args)

% Get coefficients based on order
switch args.Order
    case 2
        args.a = 1.0;
        args.b = 0.5;
    case 4
        k     =  2^(1/3) * exp(2i*pi/3);
        alpha =  1/(2-k);
        beta  = -k/(2-k);
        
        args.b = [alpha/2; alpha/2 + beta/2];
        args.a = [alpha; beta];
    otherwise
        error('Only order 2 and 4 BTSplitStepper''s are implemented.');
end

end