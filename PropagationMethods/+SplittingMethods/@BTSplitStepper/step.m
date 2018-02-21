function [y,dy,t] = step(V, x, dx, t0, varargin)
%STEP [y,dy,t] = step(V, x, dx, t0, varargin)
%{
[y,dy,t] = step(V, x, dx, t0, ... % positional args (V and x required)
'Verbose', false, 'CompDerivs', true, ... % optional positionless args
'Dcoeff', [], 'Gamma', [], 'dGamma', {});
%}

if nargin < 4
    t0 = 0;
end

if nargin < 3
    dx = {};
elseif isempty(dx)
    % convert [] --> {}
    if ~iscell(dx); dx = {}; end
elseif ~iscell(dx)
    dx = {dx}; % wrap in cell
end

% ---- Optional Input Defaults ---- %
DefaultArgs = struct( 'Verbose', false, 'CompDerivs', ~isempty(dx), ...
    'Dcoeff', [], 'Gamma', [], 'dGamma', {{}} );
p = getParser(DefaultArgs);
parse(p,varargin{:});
opts = p.Results;

% ---- Precompute, if necessary ---- %
if ~isempty(opts.Dcoeff); V = precomputeConvKernels(V,opts.Dcoeff); end
if ~isempty(opts.Gamma); V = precomputeExpDecays(V,opts.Gamma); end
if ~isempty(opts.dGamma); V = precomputeGammaDerivs(V,opts.dGamma); end

% ---- Step solution ---- %
[y,t] = BTSplitStep(V, x, t0, opts.Dcoeff, opts.Gamma, opts.Verbose);

% ---- Step derivative ---- %
if opts.CompDerivs
    switch V.Order
        % Note that all derivatives are at the same time 't' as 'y',
        % so we ignore the time output
        case 2
            [dy,~] = stepDerivOrder2(V, y, x, dx, t0, opts);
        case 4
            warning('Only order 2 derivative stepping is currently implemented; defaulting to order 2');
            [dy,~] = stepDerivOrder2(V, y, x, dx, t0, opts);
    end
else
    dy = {};
end

end

function [dy,t] = stepDerivOrder2(V, y, x, dx, t0, opts)
% [dy,t] = stepDerivOrder2(V, y, x, dx, t0, opts)

% step time one timestep forward
t = t0 + V.TimeStep;

% init dy and step derivatives
dy = cell(size(V.GammaDerivs));

for ii = 1:numel(V.GammaDerivs)
    dy{ii} = V.GammaDerivs{ii} .* y;
    if ~isempty(dx) && ~isempty(dx{ii})
        dy{ii} = dy{ii} + BTSplitStep(V, V.GammaDerivs{ii} .* x + dx{ii}, ...
            t0, opts.Dcoeff, opts.Gamma, opts.Verbose);
    else
        dy{ii} = dy{ii} + BTSplitStep(V, V.GammaDerivs{ii} .* x, ...
            t0, opts.Dcoeff, opts.Gamma, opts.Verbose);
    end
end

end

function p = getParser(DefaultArgs)

p = inputParser;
for f = fields(DefaultArgs).'
    paramName = f{1};
    defaultVal = DefaultArgs.(f{1});
    addParameter(p,paramName,defaultVal)
end

end