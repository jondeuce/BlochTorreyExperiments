function planePlot( n,p,fig,varargin )
% meant to be used in conjunction with planeFit.m
%
% Input arguments:
%   n:      the normal vector defining the plane
%   p:      a point on the plane
%
% Optional arguments:
%   xrange: vector containing desired range of x-values
%   zrange: vector containing desired range of z-values

%% code based on submission from 'Audrey', Nov 20 2012 http://stackoverflow.com/questions/13464304/how-can-i-plot-a-3d-plane-in-matlab

% Use/set showing range
if nargin<2; error('Not enough input arguments: must supply normal vector, a point on the plane'); end;
if nargin==2
    fig=1; xLim=p(1)+[-1 1]; yLim=p(2)+[-1 1];
end
figure(fig);
if nargin==3; xLim=xlim; yLim=ylim; end;
if nargin==4
    xLim=varargin{1}(:)'; yLim=ylim;
end
if nargin==5
    xLim=varargin{1}(:)';
    yLim=varargin{2}(:)';
end

[X,Y]=meshgrid(xLim,yLim);
Z=-(n(1)*(X-p(1))+n(2)*(Y-p(2)))/n(3)+p(3); % note: if normal has 0 z-component this will fail

zLim=zlim;
if max(Z(:))>max(zLim); zLim(2)=max(Z(:)); end
if min(Z(:))<min(zLim); zLim(1)=min(Z(:)); end
zlim(zLim);

reOrder = [1 2 4 3];
hold on;
patch(X(reOrder),Y(reOrder),Z(reOrder),'r');
alpha(0.3);
grid on;
hold off;

end

