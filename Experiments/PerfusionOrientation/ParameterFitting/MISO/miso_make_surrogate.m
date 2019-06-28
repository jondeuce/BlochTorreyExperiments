function rbf_fun = miso_make_surrogate(fdata, rbf_flag)
%miso_make_surrogate [rbf_fun, Data] = miso_make_surrogate(fdata, rbf_flag)
% rbf_flag = 'cub';
% rbf_flag = 'lin';
% rbf_flag = 'tps'; % Seems unstable

if nargin < 2; rbf_flag = 'cub'; end

Data.xlow       = min(fdata.xdata, [], 1); % variable lower bounds
Data.xup        = max(fdata.xdata, [], 1); % variable upper bounds
Data.dim        = size(fdata.xdata, 2); % problem dimesnion
Data.integer    = []; % indices of integer variables
Data.continuous = 1:size(fdata.xdata, 2); % indices of continuous variables
Data.S          = fdata.xdata; % sample history
Data.Y          = fdata.ydata; % function evaluation history
Data.m          = numel(fdata.ydata); % number of samples

[lambda, gamma] = rbf_params(Data, rbf_flag);
rbf_fun = @(x) rbf_prediction(x, Data, lambda, gamma, rbf_flag);

end