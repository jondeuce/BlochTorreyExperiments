function [ objval, dR2, ResultsStruct ] = perforientation_objfun( params, xdata, dR2_Data, dR2, weights, normfun, varargin )
%[ objval, dR2, ResultsStruct ] = perforientation_objfun( params, xdata, dR2_Data, weights, normfun, varargin )
% Calls perforientation_fun and returns the (weighted) residual norm.
%    xdata:    [1xN] or [MxN] row vector(s) of angle values
%    dR2_Data: [1xN] or [MxN] row vector(s) of dR2(*) data values
%    dR2:      [1xN] or [MxN] row vector(s) of simulated dR2(*) values
%    weights:  [1xN] or [MxN] row vector(s) of data weights, or string
%              indicating kind ('unit' for all ones; 'uniform' for all 1/N)
%    normfun:  [fun] function of the form @(dR2, dR2_data, weights)

if nargin < 6 || isempty(normfun); normfun = 'default'; end
if nargin < 5 || isempty(weights); weights = 'uniform'; end

% Allow for column vector inputs to be interpreted as single rows of data
if iscolumnvector(dR2_Data); dR2_Data = dR2_Data.'; end
if iscolumnvector(dR2); dR2 = dR2.'; end
if iscolumnvector(weights); weights = weights.'; end

if ischar(normfun)
    switch upper(normfun)
        case 'R2'
            normfun = @calc_R2w;
        otherwise
            if ~(strcmpi(normfun,'default') || strcmpi(normfun,'L2'))
                warning('Using default normfun: weighted L2-residual');
            end
            normfun = @calc_L2w;
    end
end

if ischar(weights)
    switch upper(weights)
        case 'UNIT' % unit weighting (dependent on number of datapoints)
            weights = ones(size(dR2_Data));
        otherwise   % uniform weighting (independent of number of datapoints)
            if ~strcmpi(weights,'uniform')
                warning('Unknown option weights = ''%s''; defaulting to ''uniform''.', weights);
            end
            weights = ones(size(dR2_Data))/size(dR2_Data, 2);
    end
end

if isempty(dR2)
    [dR2, ResultsStruct] = perforientation_fun(params, xdata, dR2_Data, varargin{:}, 'Weights', weights);
    
%     % Fake dR2 with unique minimum at params = params0 for testing
%     fakerand = @(siz) reshape( mod(1:prod(siz),pi)/pi, siz ); %deterministic ~uniformly random
%     params0 = [5; 1.0/100; 1.0/100]; % [CA; iBVF; aBVF]
%     dR2_noise = reshape( sin( fakerand([numel(dR2_Data), numel(params)]) * (params(:) - params0(:)).^2 ), size(dR2_Data) );
%     dR2 = dR2_Data + dR2_noise;
end

objval = normfun(dR2, dR2_Data, weights);

end


function [ L2w ] = calc_L2w( ymodel, ydata, w )
% Can assume that data have compatible sizes
dy2 = bsxfun(@minus, ymodel, ydata).^2; % squared diff
dy2 = bsxfun(@times, w, dy2); % weighted squared diff
L2w = sqrt(sum(dy2, 2)); % root mean square loss

% ydata = ydata(:).';
% w = w(:).';
% ymodel = reshape(ymodel, [], length(ydata));
% ydata = repmat(ydata, size(ymodel,1), 1);
% w = repmat(w, size(ymodel,1), 1);
% L2w = sqrt(sum(w.*(ymodel-ydata).^2, 2)); % sqrt(sum( w*(x-xdata)^2 ))

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

% % Standard form (note: scaling w by a constant doesn't change R2, but is
% % more convenient for calculating ybar)
% w = w(:).'./sum(w(:));
% ydata = ydata(:).';
% ybar = sum(w.*ydata);
% 
% % Reshape and replicate
% ymodel = reshape(ymodel, [], length(ydata));
% ydata = repmat(ydata, size(ymodel,1), 1);
% w = repmat(w, size(ymodel,1), 1);
% 
% % Calculate R2 using the Sum of Squares method
% % https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
% ss_residual = sum(w.*(ydata-ymodel).^2, 2);
% ss_total = sum(w.*(ydata-ybar).^2, 2);
% 
% R2w = 1 - ss_residual ./ ss_total;

end
