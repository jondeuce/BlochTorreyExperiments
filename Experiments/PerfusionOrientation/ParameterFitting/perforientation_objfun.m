function [ objval, dR2, ResultsStruct ] = perforientation_objfun( params, xdata, dR2_Data, dR2, weights, normfun, varargin )
%[ objval, dR2, ResultsStruct ] = perforientation_objfun( params, xdata, dR2_Data, weights, normfun, varargin )
% Calls perforientation_fun and returns the (weighted) residual norm.
%    xdata:    [1xN] or [MxN] row vector(s) of angle values
%    dR2_Data: [1xN] or [MxN] row vector(s) of dR2(*) data values
%    dR2:      [1xN] or [MxN] row vector(s) of simulated dR2(*) values
%    weights:  [1xN] or [MxN] row vector(s) of data weights, or string
%              indicating kind ('unit' for all ones; 'uniform' for all 1/N)
%              NOTE: these weights are for the squared residuals; pass
%                    sqrt(weights) to scale to un-squared residuals weights
%    normfun:  [fun] function of the form @(dR2, dR2_data, weights)

if nargin < 6 || isempty(normfun); normfun = 'default'; end
if nargin < 5 || isempty(weights); weights = 'uniform'; end

% Allow for column vector inputs to be interpreted as single rows of data
if iscolumnvector(dR2_Data); dR2_Data = dR2_Data.'; end
if iscolumnvector(dR2); dR2 = dR2.'; end
if iscolumnvector(weights); weights = weights.'; end

% Number of parameters
p = numel(params);

if ischar(normfun)    
    switch upper(normfun)
        case {'L2', 'L2W', 'DEFAULT'}
            normfun = @calc_L2w;
        case {'R2', 'R2W'}
            normfun = @calc_R2w;
        case {'R2ADJ', 'R2WADJ'}
            normfun = @(ymodel, ydata, w) calc_R2wadj(ymodel, ydata, w, p);
        case 'AIC'
            normfun = @(ymodel, ydata, w) calc_AIC(ymodel, ydata, w, p);
        case 'AICC'
            normfun = @(ymodel, ydata, w) calc_AICc(ymodel, ydata, w, p);
        case 'BIC'
            normfun = @(ymodel, ydata, w) calc_BIC(ymodel, ydata, w, p);
        otherwise
            if ~(strcmpi(normfun,'default') || strcmpi(normfun,'L2'))
                warning('Using default normfun: weighted L2-residual');
            end
            normfun = @calc_L2w;
    end
end

if ischar(weights)
    switch upper(weights)
        case 'UNIT' % unit weighting
            weights = ones(size(dR2_Data));
        otherwise   % uniform weighting (weights have unit sum)
            if ~strcmpi(weights,'uniform')
                warning('Unknown option weights = ''%s''; defaulting to ''uniform''.', weights);
            end
            weights = ones(size(dR2_Data))/size(dR2_Data, 2);
    end
end

if isempty(dR2)
    [dR2, ResultsStruct] = perforientation_fun(params, xdata, dR2_Data, varargin{:}, 'Weights', weights);
end

objval = normfun(dR2, dR2_Data, weights);

end

function [ res ] = calc_ResW(ymodel, ydata, w)
dy = bsxfun(@minus, ymodel, ydata); % residual
res = bsxfun(@times, sqrt(w), dy); % weighted residual
end

function [ L2w ] = calc_L2w( ymodel, ydata, w )
res = calc_ResW(ymodel, ydata, w);
L2w = sqrt(sum(res.^2, 2)); % root mean square loss
end

function [ R2w, ybar ] = calc_R2w( ymodel, ydata, w )
%CALC_R2 Calculate the (weighted) coefficient of determination R^2.
%   xdata: dataset values
%   xmodel: predicted values by model
%   weights: weights for forming weighted squared residuals

% Standard form (note: scaling w by a constant doesn't change R2, but is
% more convenient for calculating ybar)
w = bsxfun(@rdivide, w, sum(w, 2)); % [MxN] array
ybar = sum(bsxfun(@times, w, ydata), 2); % [Mx1] vector

% Calculate R2 using the Sum of Squares method
% https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
ss_residual = sum(bsxfun(@times, w, bsxfun(@minus, ydata, ymodel).^2), 2); % [Mx1] generalization of sum(w.*(ydata-ymodel).^2, 2);
ss_total = sum(bsxfun(@times, w, bsxfun(@minus, ydata, ybar).^2), 2); % [Mx1] generalization of sum(w.*(ydata-ybar).^2, 2);
R2w = 1 - ss_residual ./ ss_total;
end

function [ R2wadj ] = calc_R2wadj(ymodel, ydata, w, p)
n = size(ydata, 2);
R2w = calc_R2w(ymodel, ydata, w);
R2wadj = 1 - ((n - 1)/ (n - p)) * (1 - R2w);
end

% Calculation of information theoretic-based penalty functions
% See e.g. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2892436/)
%   Spiess, A.-N., Neumeyer, N., 2010. An evaluation of R2 as an inadequate
%   measure for nonlinear models in pharmacological and biochemical
%   research: a Monte Carlo approach.
%   BMC Pharmacol 10, 6. https://doi.org/10.1186/1471-2210-10-6

% Maximum log-likelihood of the estimated model (in the normally
% distributed error approximation)
function [ logL ] = calc_MaxLogL(ymodel, ydata, w)
n = size(ydata, 2);
xi = calc_ResW(ymodel, ydata, w);
logL = -0.5 * n * (log(2*pi) + 1 - log(n) + log(sum(xi.^2, 2)));
end

% Akaike Information Criterion
function [ AIC ] = calc_AIC(ymodel, ydata, w, p)
n = size(ydata, 2);
logL = calc_MaxLogL(ymodel, ydata, w);
AIC = 2*p - 2*logL;
end

% Bias-corrected AIC; corrects for small sample sizes
function [ AICc ] = calc_AICc(ymodel, ydata, w, p)
n = size(ydata, 2);
AIC = calc_AIC(ymodel, ydata, w, p);
AICc = AIC + 2*p*(p+1)/(n-p-1);
end

% Bayesian Information Criterion
function [ BIC ] = calc_BIC(ymodel, ydata, w, p)
n = size(ydata, 2);
logL = calc_MaxLogL(ymodel, ydata, w);
BIC = p*log(n) - 2*logL;
end


% ==== TEST CODE ==== %

% % Fake dR2 with unique minimum at params = params0 for testing
% fakerand = @(siz) reshape( mod(1:prod(siz),pi)/pi, siz ); %deterministic ~uniformly random
% params0 = [5; 1.0/100; 1.0/100]; % [CA; iBVF; aBVF]
% dR2_noise = reshape( sin( fakerand([numel(dR2_Data), numel(params)]) * (params(:) - params0(:)).^2 ), size(dR2_Data) );
% dR2 = dR2_Data + dR2_noise;

% % Print out all metrics
% if strcmpi(normfun, 'all')
%     callfun = @(weights, normfun) perforientation_objfun( ...
%         params, xdata, dR2_Data, dR2, weights, normfun, varargin{:} );
%     
%     nfuns = {'R2', 'R2w', 'L2', 'L2W', 'R2ADJ', 'R2WADJ', 'AIC', 'AICC', 'BIC', 'gibberish'};
%     for ii = 1:numel(nfuns)
%         nfun = nfuns{ii};
%         fprintf('%s: %f\n', nfun, callfun('uniform', nfun));
%         fprintf('%s: %f\n', nfun, callfun(weights, nfun));
%     end
%     
%     return
% end