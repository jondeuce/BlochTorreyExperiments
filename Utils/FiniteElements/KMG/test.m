%
%
% Tests for mesh generation with kmg2d
% J. koko, ISIMA/LIMOS, koko@isima.fr
% (c) 2011
%  
%---- Choose a domain number
% 1-  Square with holes
% 2-  Cavity
% 3-  Hook form
% 4-  Two circles
%----
domain=1;
%---- Choose a mesh size function
% 1-  Uniform
% 2-  Geometry-based
%----
sizefun=2;
switch sizefun
    case 1
        fh=@huniform;
    case 2
        fh=@hgeom;
    otherwise
        error('Unknown mesh size function.')
end

%---- Set the other mesh data
dg=1; % dg=1: 3-node triangles, 
      % dg=2: 6-node triangles
nr=0; % number of refinments

%---- Domain selection: modify the reference edge length h0 to control the
%     size of the mesh
switch domain
     case 1
        [fd,bbox,pfix]=itrou;
        h0=0.03;
        [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix);
    case 2
        [fd,bbox,pfix]=idcavity;
        h0=0.02;
        [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix);
    case 3
        [fd,bbox,pfix]=ihook;
        h0=0.015;
        [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix); 
    case 4
        [fd,bbox,pfix]=itwocircle;
        h0=0.035;
        [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix,0,0,1,-.4,0,.5);
    otherwise
        error('Unknown domain.')
end

 
% Mesh generation
%[p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix);
 
