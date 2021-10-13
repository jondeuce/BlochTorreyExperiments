function [h,x,y,z]=cylinderPlot( q,v,r,height,fig,col,alpha )
%	-   plots the right-circular cylinder defined by central-axis direction 'v',
%       a point on the central axis 'q', the radius 'r', and the height 'height'.
% 
%   -   figure is plotted on figure 'fig'
% 
%   -   note:   notation is consistent with [q,v,r]=lscylinder(...)
%               from lscylinder.m

if nargin < 7; alpha = 0.2; end
if nargin < 6; col = 'b'; end
if nargin < 5 || isempty(fig); fig = figure; end

axis image
[hh,xx,yy,zz]=Cylinder(q-height/2*v,q+height/2*v,r,20,col,false,false,alpha);

% view([0.5852    0.7752    0.2380]);
% length=50; baseangle=60; tipangle=30; width=20;
% properties={length,baseangle,tipangle,width};
% arrow(q-height/2*v,q+height/2*v,properties{:},'FaceColor','c');

if nargout>0; h=hh; x=xx; y=yy; z=zz; end

end

