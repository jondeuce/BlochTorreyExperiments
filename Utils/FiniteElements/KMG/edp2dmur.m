function [p,t,bpx,bpy]=edp2dmur(ax,bx,ay,by,nx,ny)
%EDP2DMUR Maillage uniforme d'un rectangle par des triangles
%--------------------------------------------------------------------
% Synopsis:
%      [p,t]=edp2drgd(ax,bx,ay,by,nx,ny)
%      [p,t,bpx,bpy]=edp2drgd(ax,bx,ay,by,nx,ny)
% Arguments:
%   (ax,bx) - intervalle des x
%   (ay,by) - intervalle des y
%        nx - nombre de points en x
%        ny - nombre de points en y
%        p  - coordonnees des sommets, tableau np*2
%        t  - sommets des triangles, tableau nt*3
%  bpx,bpy  - noeuds du bord du rectangle, tableaux nx*2 et ny*2
%             bpx(:,1), bpx(:,2), noeuds de (ax,bx)*{ay} et (ax,bx)*{by}
%             bpy(:,1), bpy(:,2), noeuds de {ax}*(ay,by) et {bx}*(ay,by)
%--------------------------------------------------------------------
np=nx*ny;
nq=(nx-1)*(ny-1);
nt=2*nq;

% coordonnées des sommets
hx=(bx-ax)/(nx-1); hy=(by-ay)/(ny-1);
[x,y]=meshgrid(ax:hx:bx,ay:hy:by);
xx=x'; yy=y'; p=[xx(:),yy(:)];

% sommets des quadrangles
ip=[1:nx*ny]';
ib1=[1:nx]'; ib2=nx*[1:ny]';
ib3=[nx*(ny-1)+1:nx*ny]'; ib4=[1:nx:nx*ny]';
ib23=union(ib2,ib3); ib34=union(ib3,ib4);
ib14=union(ib1,ib4); ib12=union(ib1,ib2);
iq1=setdiff(ip,ib23); iq2=setdiff(ip,ib34);
iq3=setdiff(ip,ib14); iq4=setdiff(ip,ib12);

% triangulation
t=zeros(nt,3);
t(1:nq,1)=iq1;      t(1:nq,2)=iq2;      t(1:nq,3)=iq3;
t(nq+1:2*nq,1)=iq3; t(nq+1:2*nq,2)=iq4; t(nq+1:2*nq,3)=iq1;

if (nargin==2)
   return
end
% sommets du bord
bpx=[ib1 ib3];
bpy=[ib4 ib2];
