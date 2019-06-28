function allfigs = miso_plot_surrogate(fdata, rbf_flag)
%miso_plot_surrogate allfigs = miso_plot_surrogate(fdata, rbf_flag)

if nargin < 2
    rbf_flag = 'cub';
end

rbf_fun = miso_make_surrogate(fdata, rbf_flag);

npts = 250;
range = @(x) linspace(min(x), max(x), npts);
minstr = ['Min = ', num2str(fdata.ybest)];
titlefun = @(x,y,fix) title(['RBF AICc vs. ', x, ' and ', y, ' (', fix, ' fixed; ', minstr, ')']);
plotargs = {'EdgeColor', 'None'};
scatterargs = {50 * ones(size(fdata.ydata)), 'ko', 'filled'};

x = fdata.iBVF; y = fdata.aBVF; z = fdata.CAbest * ones(size(x));
[X,Y] = meshgrid(range(x), range(y));
Z = fdata.CAbest * ones(size(X));
F = reshape(rbf_fun([X(:), Y(:), Z(:)]), size(Z));
fig1 = figure; surf(X,Y,F,plotargs{:});
hold on; scatter3(x, y, rbf_fun([x,y,z]),scatterargs{:})
titlefun('iBVF', 'aBVF', 'CA'); xlabel('iBVF'); ylabel('aBVF'); zlabel('RBF AICc');

x = fdata.aBVF; y = fdata.CA; z = fdata.iBVFbest * ones(size(x));
[X,Y] = meshgrid(range(x), range(y));
Z = fdata.iBVFbest * ones(size(X));
F = reshape(rbf_fun([Z(:), X(:), Y(:)]), size(Z));
fig2 = figure; surf(X,Y,F,plotargs{:});
hold on; scatter3(x, y, rbf_fun([z,x,y]),scatterargs{:})
titlefun('aBVF', 'CA', 'iBVF'); xlabel('aBVF'); ylabel('CA'); zlabel('RBF AICc');

x = fdata.iBVF; y = fdata.CA; z = fdata.aBVFbest * ones(size(x));
[X,Y] = meshgrid(range(x), range(y));
Z = fdata.aBVFbest * ones(size(X));
F = reshape(rbf_fun([X(:), Z(:), Y(:)]), size(Z));
fig3 = figure; surf(X,Y,F,plotargs{:});
hold on; scatter3(x, y, rbf_fun([x,z,y]),scatterargs{:})
titlefun('iBVF', 'CA', 'aBVF'); xlabel('iBVF'); ylabel('CA'); zlabel('RBF AICc');

if nargout >= 1
    allfigs = [fig1; fig2; fig3];
end

end