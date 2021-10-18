function [p,t,be,bn]=kmg2dref(p0,t0,dg,fd,deps,tol,varargin)
%KMG2DREF Refine a two-dimensional triangular mesh
%--------------------------------------------------------------------
% [p,t,be,bn]=kmg2dref(p0,t0,dg,fd,deps,tol)
% [p,t,be,bn]=kmg2dref(p0,t0,2)
% Input:
%   p0       : Initial node coordinates np0*2
%   t0       : Initial triangle vertices nt0*3
%   dg       : Type of the new triangulation: 
%              dg=1 3-node; dg=2 6-node
%   fd       : Signed distance function d(x,y) 
%   deps     : Finite difference spacing in MATLAB function gradient
%   tol      : Tolerance of the Newton-type iteration 
% Output:
%   p        : Nodes coordinates of the refined mesh, np*2
%   t        : Triangle nodes of the refined mes, nt*3 or nt*6
%   be       : Boundary edgdes in the refined mesh, ne*2 or ne*3 
%   bn       : Boundary nodes in the refined mesh, nb*1   
%--------------------------------------------------------------------
% (c) 2009, Koko J., ISIMA, koko@isima.fr
%--------------------------------------------------------------------
%
np=size(p0,1); nt=size(t0,1);
 
% Extract all edges
[e,ib,je]=kmg2dedg(t0);

% Form new nodes (mid-point of edges)
ne=size(e,1);  p=[p0; (p0(e(:,1),:)+p0(e(:,2),:))/2];

% Form new edges
ip1=t0(:,1); ip2=t0(:,2); ip3=t0(:,3);
pm=[(np+1):(np+ne)]'; lmp=pm(je);
mp1=lmp(1:nt); mp2=lmp(nt+1:2*nt); mp3=lmp(2*nt+1:3*nt);

% Form new triangles
if (dg==1)
   t=zeros(4*nt,3);
   t(1:nt,:)=[ip1 mp1 mp3];
   t((nt+1):2*nt,:)=[mp1 ip2 mp2];
   t((2*nt+1):3*nt,:)=[mp2 ip3 mp3];
   t((3*nt+1):4*nt,:)=[mp1 mp2 mp3];
   
   % Force new boundary nodes
   pb=p(np+ib,:); 
   [p(np+ib,:),iterb]=pnodes(pb,fd,deps,tol,varargin{:});
elseif (dg==2)
   t=zeros(nt,6); t(:,1:3)=t0;
   t(:,4:6)=[mp1 mp2 mp3];
end

% Extract (from the new mesh) boundary edges & nodes
[e,ib]=kmg2dedg(t);
be=e(ib,:);
bn=unique(be);
 
