function ellipsoidPlot( r0,r,fig,varargin )
% plots the ellipsoid associated with the vectors 'r0' and 'r' representing
% the coordinates of the center of the ellipse, and the radii in the x,y,z
% directions, respectively

figure(fig); hold on;
ellipsoid(r0(1),r0(2),r0(3),r(1),r(2),r(3));
hold off; axis image; hidden off;

end

