function allfigs = miso_plot_surrogate(Data, rbf_flag)
%miso_plot_surrogate allfigs = miso_plot_surrogate(fdata, rbf_flag)

if nargin < 2; rbf_flag = Data.surrogate; end

if strcmp(rbf_flag, 'rbf_c') %cubic RBF
    rbf_flag = 'cub';
elseif strcmp(rbf_flag, 'rbf_l') %linear RBF
    rbf_flag = 'lin';
elseif strcmp(rbf_flag, 'rbf_t') %thin plate spline RBF
    rbf_flag = 'tps';
else
    warning('rbf type ''%s'' not defined; defaulting to ''lin''', rbf_flag);
    rbf_flag = 'lin';
end

rbf_fun = miso_make_surrogate(Data, rbf_flag);

% Make sure S samples are unique (e.g. due to rounding in printing)
[~, ix, ~] = unique(Data.S, 'rows');
Data.S = Data.S(ix, :);
Data.Y = Data.Y(ix, :);

% Sort by yvalue in appropriate order
[~, iy] = sort(Data.Y, 'descend');
Data.S = Data.S(iy, :);
Data.Y = Data.Y(iy, :);

% Get and make fields
xfields = Data.xfields;
xfieldnames = Data.xfieldnames;
yfieldnames = Data.yfieldnames;
xfieldnamesbest = strcat(xfieldnames, 'best');
yfieldnamesbest = strcat(yfieldnames, 'best');

for ii = 1:numel(xfields)
    Data.(xfieldnames{xfields(ii)}) = Data.S(:, xfields(ii));
    Data.(xfieldnamesbest{xfields(ii)}) = Data.S(end, xfields(ii));
end
Data.(yfieldnames{1}) = Data.Y;
Data.(yfieldnamesbest{1}) = Data.Y(end);

% Plot surface slices
npts = 250;
range = @(x) linspace(min(x), max(x), npts);
minstr = ['Min = ', num2str(Data.(yfieldnamesbest{1}))];
titlefun = @(x,y,fix) title(['RBF ', yfieldnames{1}, ' vs. ', x, ' and ', y, ' (', fix, ' fixed; ', minstr, ')']);
plotargs = {'EdgeColor', 'None'};
scattersurfaceargs = {30, 'ko', 'filled'};

x = Data.(xfieldnames{1}); y = Data.(xfieldnames{2}); z = Data.(xfieldnamesbest{3}) * ones(size(x));
w = []; if numel(xfields) > 3; w = Data.(xfieldnamesbest{4}) * ones(size(x)); end
f = rbf_fun([x,y,z,w]);
[X,Y] = meshgrid(range(x), range(y));
Z = Data.(xfieldnamesbest{3}) * ones(size(X));
W = []; if numel(xfields) > 3; W = Data.(xfieldnamesbest{4}) * ones(size(X)); end
F = reshape(rbf_fun([X(:), Y(:), Z(:), W(:)]), size(Z));
zl = [min(min(f(:)), min(F(:))), max(max(f(:)), max(F(:)))];
fig1 = figure; surf(X,Y,F,plotargs{:}); hold on; grid minor;
scatter3(x, y, f + 0.005 * diff(zl), scattersurfaceargs{:});
xlim([min(x), max(x)]); ylim([min(y), max(y)]); zlim(zl);
titlefun(xfieldnames{1}, xfieldnames{2}, xfieldnames{3}); xlabel(xfieldnames{1}); ylabel(xfieldnames{2}); zlabel(['RBF ', yfieldnames{1}]);

x = Data.(xfieldnames{2}); y = Data.(xfieldnames{3}); z = Data.(xfieldnamesbest{1}) * ones(size(x)); w = [];
w = []; if numel(xfields) > 3; w = Data.(xfieldnamesbest{4}) * ones(size(x)); end
f = rbf_fun([z,x,y,w]);
[X,Y] = meshgrid(range(x), range(y));
Z = Data.(xfieldnamesbest{1}) * ones(size(X));
W = []; if numel(xfields) > 3; W = Data.(xfieldnamesbest{4}) * ones(size(X)); end
F = reshape(rbf_fun([Z(:), X(:), Y(:), W(:)]), size(Z));
zl = [min(min(f(:)), min(F(:))), max(max(f(:)), max(F(:)))];
fig2 = figure; surf(X,Y,F,plotargs{:}); hold on; grid minor;
scatter3(x, y, f + 0.005 * diff(zl), scattersurfaceargs{:});
xlim([min(x), max(x)]); ylim([min(y), max(y)]); zlim(zl);
titlefun(xfieldnames{2}, xfieldnames{3}, xfieldnames{1}); xlabel(xfieldnames{2}); ylabel(xfieldnames{3}); zlabel(['RBF ', yfieldnames{1}]);

x = Data.(xfieldnames{1}); y = Data.(xfieldnames{3}); z = Data.(xfieldnamesbest{2}) * ones(size(x));
w = []; if numel(xfields) > 3; w = Data.(xfieldnamesbest{4}) * ones(size(x)); end
f = rbf_fun([x,z,y,w]);
[X,Y] = meshgrid(range(x), range(y));
Z = Data.(xfieldnamesbest{2}) * ones(size(X));
W = []; if numel(xfields) > 3; W = Data.(xfieldnamesbest{4}) * ones(size(X)); end
F = reshape(rbf_fun([X(:), Z(:), Y(:), W(:)]), size(Z));
zl = [min(min(f(:)), min(F(:))), max(max(f(:)), max(F(:)))];
fig3 = figure; surf(X,Y,F,plotargs{:}); hold on; grid minor;
scatter3(x, y, f + 0.005 * diff(zl), scattersurfaceargs{:});
xlim([min(x), max(x)]); ylim([min(y), max(y)]); zlim(zl);
titlefun(xfieldnames{1}, xfieldnames{3}, xfieldnames{2}); xlabel(xfieldnames{1}); ylabel(xfieldnames{3}); zlabel(['RBF ', yfieldnames{1}]);

if nargout >= 1
    allfigs = [fig1; fig2; fig3];
end

end