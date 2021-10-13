function [H,Fig] = rectpatchPlot( m, fig )
%RECTPRISMPLOT plots the rectangular prism defined by extent m
% 
% input:
%   -'m': a 2x3 matrix containing the minimum values [xmin, ymin, zmin] of
%   the extent of the prism in the first row, and the maximum values of the
%   extend [xmax, ymax, zmax] in the second row
% 
%       e.g. if you have an Nx3 point cloud 'X', the axis-aligned bounding
%       box (AABB) of the cloud could be plotted with rectprismPlot(m),
%       with m = [ max(X,[],1); min(X,[],1 ]

x=[0 1 1 0 0 0;1 1 0 0 1 1;1 1 0 0 1 1;0 1 1 0 0 0]*(m(2,1)-m(1,1))+m(1,1);
y=[0 0 1 1 0 0;0 1 1 0 0 0;0 1 1 0 1 1;0 0 1 1 1 1]*(m(2,2)-m(1,2))+m(1,2);
z=[0 0 0 0 0 1;0 0 0 0 0 1;1 1 1 1 0 1;1 1 1 1 0 1]*(m(2,3)-m(1,3))+m(1,3);

if nargin < 2; fig = figure; end

h = zeros(6,1);
for ii = 1:6
    h(ii) = patch(gca(fig), x(:,ii), y(:,ii), z(:,ii), 'b');
end
set(h, 'edgecolor', 'k', 'facealpha', 0);

if nargout>0; H=h; end
if nargout>1; Fig=fig; end

end

