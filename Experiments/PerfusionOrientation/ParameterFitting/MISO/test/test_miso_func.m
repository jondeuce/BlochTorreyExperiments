function Data = test_miso_func(rbf_flag)
%MISO_CALL_FUN Data = test_miso_func
% Example optimization problem using `perfusionorientation_fun` samples.

if nargin < 1
    rbf_flag = 'lin';
end

Data = test_func_eval_data();
Data.xlow = min(Data.S, [], 1); % variable lower bounds
Data.xup  = max(Data.S, [], 1); % variable upper bounds

rbf_fun = miso_make_surrogate(Data, rbf_flag);
Data.objfunction = @(x) miso_call_fun(rbf_fun, x); % handle to objective function

miso_plot_surrogate(Data, rbf_flag);

end