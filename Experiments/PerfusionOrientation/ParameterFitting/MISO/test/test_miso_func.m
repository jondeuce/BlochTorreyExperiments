function Data = test_miso_func
%MISO_CALL_FUN Data = test_miso_func
% Example optimization problem using `perfusionorientation_fun` samples.

fdata = load('func_eval_data');
fdata = fdata.func_eval_data;
xdata = fdata.xdata; % [fdata.iBVF, fdata.aBVF, fdata.CA];
ydata = fdata.ydata; % fdata.AICc;

% Sample points were rounded; keep only unique points
[xdata, ix, ~] = unique(xdata, 'rows');
ydata = ydata(ix);
[ydata, iy] = sort(ydata, 'descend');
xdata = xdata(iy, :);

Data.xlow = min(xdata, [], 1); % variable lower bounds
Data.xup  = max(xdata, [], 1); % variable upper bounds
Data.dim = 3; % problem dimesnion
Data.integer = []; % indices of integer variables
Data.continuous = [1,2,3]; % indices of continuous variables

fun = rbf_func_factory(Data, fdata, xdata, ydata);
Data.objfunction = @(x) miso_call_fun(fun, x); % handle to objective function

end %function

function rbf_func = rbf_func_factory(Data, fdata, xdata, ydata)

Data.S = xdata; %set sample history
Data.Y = ydata; %set function evaluation history
Data.m = size(xdata,1); %set number of samples

rbf_flag = 'cub';
% rbf_flag = 'lin';
% rbf_flag = 'tps'; % Seems unstable
[lambda, gamma] = rbf_params(Data, rbf_flag);
rbf_func = @(x) rbf_prediction(x, Data, lambda, gamma, rbf_flag);

plotargs = {'EdgeColor', 'None'};
npts = 250;

[X,Y] = meshgrid( ...
    linspace(min(fdata.iBVF), max(fdata.iBVF), npts), ...
    linspace(min(fdata.aBVF), max(fdata.aBVF), npts));
Z = fdata.CAbest * ones(size(X));
F = reshape(rbf_func([X(:), Y(:), Z(:)]), size(Z));
figure, surf(X,Y,F,plotargs{:});
title('CA Fixed'); xlabel('iBVF'); ylabel('aBVF'); zlabel('RBF');

[X,Y] = meshgrid( ...
    linspace(min(fdata.aBVF), max(fdata.aBVF), npts), ...
    linspace(min(fdata.CA), max(fdata.CA), npts));
Z = fdata.iBVFbest * ones(size(X));
F = reshape(rbf_func([Z(:), X(:), Y(:)]), size(Z));
figure, surf(X,Y,F,plotargs{:});
title('iBVF Fixed'); xlabel('aBVF'); ylabel('CA'); zlabel('RBF');

[X,Y] = meshgrid( ...
    linspace(min(fdata.iBVF), max(fdata.iBVF), npts), ...
    linspace(min(fdata.CA), max(fdata.CA), npts));
Z = fdata.aBVFbest * ones(size(X));
F = reshape(rbf_func([X(:), Z(:), Y(:)]), size(Z));
figure, surf(X,Y,F,plotargs{:});
title('aBVF Fixed'); xlabel('iBVF'); ylabel('CA'); zlabel('RBF');

end