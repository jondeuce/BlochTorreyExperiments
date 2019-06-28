function Data = test_miso_func
%MISO_CALL_FUN Data = test_miso_func
% Example optimization problem using `perfusionorientation_fun` samples.

fdata = test_func_eval_data('miso');
Data.xlow = min(fdata.xdata, [], 1); % variable lower bounds
Data.xup  = max(fdata.xdata, [], 1); % variable upper bounds
Data.dim  = size(fdata.xdata, 2); % problem dimesnion
Data.integer    = []; % indices of integer variables
Data.continuous = 1:size(fdata.xdata, 2); % indices of continuous variables

rbf_fun = miso_make_surrogate(fdata, 'cub');
Data.objfunction = @(x) miso_call_fun(rbf_fun, x); % handle to objective function

miso_plot_surrogate(fdata, 'cub');

end