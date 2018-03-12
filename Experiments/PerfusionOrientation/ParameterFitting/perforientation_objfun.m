function [ objval ] = perforientation_objfun( params, xdata, dR2_Data, dR2, weights, normfun, varargin )
%[ objval ] = perforientation_objfun( params, xdata, dR2_Data, weights, normfun, varargin )
% Calls perforientation_fun and returns the (weighted) residual norm.

if isempty(normfun) || (ischar(normfun) && strcmpi(normfun, 'default'))
    normfun = @(x,xdata,w) sqrt(sum(vec(w).*vec(x-xdata).^2));
end
if isempty(weights); weights = 'uniform'; end

if ischar(weights)
    switch upper(weights)
        case 'UNIT' % unit weighting (dependent on number of datapoints)
            weights = ones(size(dR2_Data));
        otherwise   % uniform weighting (independent of number of datapoints)
            if ~strcmpi(weights,'uniform')
                warning('Unknown option weights = ''%s''; defaulting to ''uniform''.', weights);
            end
            weights = ones(size(dR2_Data))/numel(dR2_Data);
    end
end

if isempty(dR2)
    dR2 = perforientation_fun(params, xdata, dR2_Data, 'Weights', weights, varargin{:});
    
    % Fake dR2 with unique minimum at params = params0 for testing
    %     fakerand = @(siz) reshape( mod(1:prod(siz),pi)/pi, siz ); %deterministic ~uniformly random
    %     params0 = [5; 1.25/100; 0.5/100]; % [CA; iBVF; aBVF]
    %     dR2_noise = reshape( sin( fakerand([numel(dR2_Data), numel(params)]) * (params(:) - params0(:)).^2 ), size(dR2_Data) );
    %     dR2 = dR2_Data + dR2_noise;
end

objval = normfun(dR2, dR2_Data, weights);

end
