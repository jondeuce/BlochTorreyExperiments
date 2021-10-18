function [Fig] = plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, titlestr, fig, col, alpha, verb )
%PLOT_CYLINDERS_IN_BOX Plots the cylinderes defined by the [3xN] set of
%points and directinos p and v and [1xN] radii r in the box defined by the
%[3x1] vectors BoxDims and BoxCenter.

if nargin < 10 || isempty(verb);     verb = false; end
if nargin < 9  || isempty(alpha);    alpha = 0.2; end
if nargin < 8  || isempty(col);      col = 'b'; end
if nargin < 7  || isempty(fig);      fig = figure; end
if nargin < 6  || isempty(titlestr); titlestr = ''; end

BoxBounds = [ BoxCenter(:)' - 0.5*BoxDims(:)'
              BoxCenter(:)' + 0.5*BoxDims(:)' ];
[~, fig] = rectpatchPlot( BoxBounds, fig );

ii_print = round(linspace(1,size(p,2),11)); % Print 0%, 10%, ..., 100%
print_count = 1;
for ii = 1:size(p,2)
    if verb && ii == ii_print(print_count)
        fprintf('Plotting cylinders... (%d%% complete)\n', round(100*ii/size(p,2)));
        print_count = print_count + 1;
    end
    % Compute the midpoint and height of the cylinder to be plotted
    [tmin, tmax, ~, ~, pmid] = rayBoxIntersection( p(:,ii), v(:,ii), BoxDims, BoxCenter );
    if ~isnan(tmin) && ~isnan(tmax)
        cylheight = tmax - tmin;
        cylinderPlot( pmid, v(:,ii), r(ii), cylheight, fig, col, alpha );    
    end
end

axis image
DilatationFact = 1 + 1e-2;
DilatedBoxBounds = bsxfun(@plus, BoxCenter, DilatationFact * bsxfun(@minus, BoxBounds, BoxCenter));
axis(DilatedBoxBounds(:)');

xlabel('x'); ylabel('y'); zlabel('z');
if ~isempty( titlestr ); title( titlestr ); end

% Should avoid drawnow here in case this function is called many times in
% an inner loop, e.g. to plot many different cylinders
% drawnow

if nargout >= 1; Fig = fig; end

end
