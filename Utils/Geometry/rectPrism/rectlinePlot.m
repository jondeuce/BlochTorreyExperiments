function h = rectlinePlot( m, fig )
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

X=[ [m(1,1); m(1,1); m(1,1); m(2,1); m(2,1); m(2,1); m(2,1); m(2,1); m(1,1); m(1,1); m(2,1); m(2,1)]...
    [m(2,1); m(1,1); m(1,1); m(1,1); m(2,1); m(2,1); m(1,1); m(2,1); m(1,1); m(1,1); m(1,1); m(2,1)] ];

Y=[ [m(1,2); m(1,2); m(1,2); m(2,2); m(2,2); m(2,2); m(1,2); m(1,2); m(2,2); m(2,2); m(2,2); m(2,2)]...
    [m(1,2); m(2,2); m(1,2); m(2,2); m(1,2); m(2,2); m(1,2); m(1,2); m(1,2); m(2,2); m(2,2); m(1,2)] ];

Z=[ [m(1,3); m(1,3); m(1,3); m(2,3); m(2,3); m(2,3); m(2,3); m(2,3); m(2,3); m(2,3); m(1,3); m(1,3)]...
    [m(1,3); m(1,3); m(2,3); m(2,3); m(2,3); m(1,3); m(2,3); m(1,3); m(2,3); m(1,3); m(1,3); m(1,3)] ];

figure(fig); hold on;
hh=zeros(12,1);
for i=1:12
    hh(i)=line(X(i,:),Y(i,:),Z(i,:));
end
hold off;

if nargout>0; h=hh; end;

end

