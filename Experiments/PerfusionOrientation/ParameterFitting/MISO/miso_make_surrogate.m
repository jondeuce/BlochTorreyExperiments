function rbf_fun = miso_make_surrogate(fdata, rbf_flag)
%miso_make_surrogate [rbf_fun, Data] = miso_make_surrogate(fdata, rbf_flag)
% rbf_flag = 'lin';
% rbf_flag = 'cub'; % Seems unstable
% rbf_flag = 'tps'; % Seems unstable

if nargin < 2; rbf_flag = 'lin'; end

Data.xlow       = min(fdata.S, [], 1); % variable lower bounds
Data.xup        = max(fdata.S, [], 1); % variable upper bounds
Data.dim        = fdata.dim; % problem dimesnion
Data.integer    = fdata.integer; % indices of integer variables
Data.continuous = fdata.continuous; % indices of continuous variables
Data.S          = fdata.S; % sample history
Data.Y          = fdata.Y; % function evaluation history
Data.m          = numel(fdata.Y); % number of samples

[lambda, gamma] = rbf_params(Data, rbf_flag);
rbf_fun = @(x) rbf_prediction(x, Data, lambda, gamma, rbf_flag);

end