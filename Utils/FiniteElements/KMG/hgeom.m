function fh=hgeom(p,fd,pmax,varargin)
%HGEOM - Mesh size function based on medial axis distance
%--------------------------------------------------------------------
% Inputs:
%    p     : points coordinates
%    fd    : signed distance function
%    pmax  : approximate medial axis points
%
% output:
%    fh    : mesh size function of points p
%--------------------------------------------------------------------
%  (c) 2011, Koko J., ISIMA, koko@isima.fr
%--------------------------------------------------------------------

% Edge lengths ratio constant
alpha=0.4;

% Normalized signed distance fuction
fh1=feval(fd,p,varargin{:}); fh1=fh1/max(abs(fh1));

% Normalized medial axis distance
fh2=distp(p,pmax); fh2=fh2/max(fh2);

% Size function
fh=alpha+abs(fh1)+fh2; 
%
%---- Medial axis distance function ---------------------------------
%
function [d,id]=distp(p,pp)
% Compute the distance between p(i) and the set of point pp
  np=size(p,1); npp=size(pp,1);
  [xp,xpp]=meshgrid(p(:,1),pp(:,1));
  [yp,ypp]=meshgrid(p(:,2),pp(:,2));
  dm=sqrt((xp-xpp).^2+(yp-ypp).^2);
  if (nargout ==1 )
      d=min(dm); d=d';
  else
      [d,id]=min(dm); d=d'; id=id';
  end