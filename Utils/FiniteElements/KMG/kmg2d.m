function [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix,varargin)
%KMG2D 2D-mesh generator using signed distance & size functions.
%--------------------------------------------------------------------
% [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr)
% [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix)
% [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,[],fparam)
% [p,t,be,bn]=kmg2d(fd,fh,h0,bbox,dg,nr,pfix,fparam)
% Input:
%   fd       : Signed distance  d(x,y)  (function handle)
%   fh       : Mesh size function h(x,y) (function handle)
%   h0       : Reference edge length
%   bbox     : Boundary box [xmin,ymin; xmax,ymax]
%   dg       : Type of triangular mesh: dg=1 linear; dg=2 quadratic
%   nr       : Number of refinements (nr=0, without refinement)
%   pfix     : Fixed nodes (nodes that must appear in the mesh)
%   varargin : Additional parameters passed to fd and fh (optional)
% Output:
%   p        : Node coordinates np*2
%   t        : Triangle vertices nt*3 (linear) or nt*6 (quadratic)
%   be       : Boundary edges    ne*2 (linear) or nt*3 (quadratic)
%   bn       : Boundary nodes    nb*1
%--------------------------------------------------------------------
% (c) 2009, Koko J., ISIMA, koko@isima.fr
%--------------------------------------------------------------------
% Scale factor and time step
Fscale=1.2; dt=.1;
% Convergence tolerances
epsp=.005; epsr=.5; deps=sqrt(eps)*h0; geps=.001*h0;
iterMax=5000; mp=5;
% Minimum triangle quality
qmin=.5;

% Initialize the fix points set
if (nargin>=7)
    npf=size(pfix,1);
else
    pfix=[]; npf=0; 
end

% Extract mesh size function name and compare strings
fhn=func2str(fh);

% Compute the approximate medial axis
if strcmpi(fhn,'hgeom') 
   hh0=h0/2;
   ex=[bbox(1,1):hh0:bbox(2,1)]; ey=[bbox(1,2):hh0:bbox(2,2)];
   [x,y]=meshgrid(ex,ey);
   z=feval(fd,[x(:) y(:)],varargin{:}); 
   [mx,nx]=size(x); z=reshape(z,mx,nx);
   [zx,zy]=gradient(z,hh0);  zz=sqrt(zx.*zx+zy.*zy); 
   imax=find(zz<.99 & z<=0); pmax=[x(imax) y(imax)]; pmax=[pmax; pfix];
   clear ex ey x y z zx zy zz imax
end

% Initial grid
[x,y]=meshgrid(bbox(1,1):h0:bbox(2,1),bbox(1,2):h0*sqrt(3)/2:bbox(2,2));
x(2:2:end,:)=x(2:2:end,:)+h0/2;                      
p=[x(:),y(:)];   
clear x y

% Remove points outside the region, apply the rejection method
p=p(feval(fd,p,varargin{:})<geps,:); 
 
if strcmpi(fhn,'hgeom') 
   r0=feval(fh,p,fd,pmax,varargin{:});
else
   r0=feval(fh,p,varargin{:});   
end

r0=1./r0.^2;                    
p=[pfix; p(rand(size(p,1),1)<r0./max(r0),:)];
clear r0

% Remove nodes outside the domain
np0=size(p,1); dp=feval(fd,p,varargin{:});
ii=find(dp<geps); q=p(ii,:);

% Add fixed nodes
if (npf>0), p=setdiff(q,pfix,'rows'); p=[pfix; p];
else, p=q; end
clear q dp

% Initial distribution
close
plot(p(:,1),p(:,2),'.')
fprintf('\n Initial number of nodes : %5d \n\n',size(p,1))
pause(3)

itri=0;
iter=0; ifix=[];  tolp=1; tolr=10^2; 
while (iter<iterMax & tolp>epsp)
  iter=iter+1;

  % Delaunay triangluation
  if (tolr>epsr)
    np=size(p,1); itri=1;
    p0=p;
    t=delaunayn(p);
    % Reject triangles with centroid outside the domain
    pm=(p(t(:,1),:)+p(t(:,2),:)+p(t(:,3),:))/3;
    t=t(feval(fd,pm,varargin{:})<-geps,:);
    % Reorder (locally) triangle vertices counter clockwise
    ar=tarea(p,t); it=find(ar<0);
    itt=t(it,2); t(it,2)=t(it,3); t(it,3)=itt;
    % Form all edges without duplication & extract boundary nodes 
    [e,ib]=kmg2dedg(t);
    be=e(ib,:);
    bn=unique(be);
    ii=setdiff([1:np],bn);
    % Graphical output of the current mesh
    triplot(t,p(:,1),p(:,2)), axis equal,axis off,drawnow, hold off
  end
  
  % Compute edge lengths & forces
  evec=p(e(:,1),:)-p(e(:,2),:);
  Le=sqrt(sum(evec.^2,2));
  if (strcmpi(fhn,'hgeom'))
     He=feval(fh,(p(e(:,1),:)+p(e(:,2),:))/2,fd,pmax,varargin{:});
  else
     He=feval(fh,(p(e(:,1),:)+p(e(:,2),:))/2,varargin{:});
  end
  L0=He*Fscale*sqrt(sum(Le.^2)/sum(He.^2));
  L=Le./L0;
  
  % split too long edges
  if (iter>np)
      il=find(L>1.5); 
      if (length(il)>0)
          p=[p; (p(e(il,1),:)+p(e(il,2),:))/2];
          tolp=1; tolr=1.2; 
          fprintf('Number of edges split : %3d \n',length(il))
          continue
      end
  end
  
  F=(1-L.^4).*exp(-L.^4)./L;
  Fvec=F*[1,1].*evec;
 
  % Assemble edge forces on nodes
  Fe=full(sparse(e(:,[1,1,2,2]),ones(size(F))*[1,2,1,2],[Fvec,-Fvec],np,2));
  Fe(1:npf,:)=0;
  if (iter>mp*np), Fe(ifix,:)=0; end
 
  % Move nodes
   p=p+dt*Fe;  
  
  % Project external nodes onto the boundary
  pb=p(bn,:); [p(bn,:),iterb]=pnodes(pb,fd,deps,1000*deps,varargin{:});
  dp=feval(fd,p,varargin{:}); 
  
  % Stopping criteria
  dp=dt*sqrt(sum(Fe.^2,2));
  tolp=max(dp(ii))/h0;
  tolr=max(sqrt(sum((p-p0).^2,2))/h0);
  
  % Check the nodes speed if iter>mp*np
  if (iter>mp*np), ifix=find(sqrt(sum(dp.^2,2))<dt*epsp); end
  
  % check the triangle orientation & quality if tolp<epsp 
  if (tolp<epsp)
     [ar,qt,te]=tarea(p,t); [qtmin,itmin]=min(qt);
     if (min(ar)<0 | qtmin < qmin)
         tolp=1; tolr=1.2; ifix=[];
         if (qtmin < qmin)
            it=t(itmin,:); 
            it=setdiff(it,union([1:npf]',ifix)); pt=p(it,:);
            p(it,:)=[];
            if (length(it)==3)
                p=[p; (pt(1,:)+pt(2,:)+pt(3,:))/3];
            elseif (length(it)==2)
                p=[p; (pt(1,:)+pt(2,:))/2];
            end
         end
     end
     if (min(qt)<qmin)
        fprintf('Low quality triangle qt=%12.6f \n',min(qt))
     end
  end
   fprintf('iteration: %4d  triangulation: %2d  tolerance: %10.6f   Fixed nodes: %4d\n',...
             iter,itri,tolp,length(ifix)-npf)
  itri=0;
end

% Refine the mesh if necessary
if (dg > 1  | nr > 1)
  if (dg == 1),
     for i=1:nr, [p,t,be,bn]=kmg2dref(p,t,dg,fd,deps,1000*deps,varargin{:}); end
  else 
     for i=1:nr-1, [p,t,be,bn]=kmg2dref(p,t,1,fd,deps,1000*deps,varargin{:}); end
     [p,t,be,bn]=kmg2dref(p,t,2,fd,varargin{:});
  end
end

% Plot the final mesh
close
triplot(t,p(:,1),p(:,2)), axis equal,axis off,drawnow, hold off
fprintf('\nNumber of nodes ---------: %5d \n',size(p,1))
fprintf('Number of triangles--------: %5d \n',size(t,1))
fprintf('Triangle quality measure---> %5.3f \n',min(qt))
fprintf('Number of iterations-------: %4d \n',iter)
end
 

%
%---- Triangles area & quality --------------------------------------------
%
function [ar,qt,te]=tarea(p,t)
%Compute triangle area and quality
%
it1=t(:,1); it2=t(:,2); it3=t(:,3);
x21=p(it2,1)-p(it1,1); y21=p(it2,2)-p(it1,2); 
x31=p(it3,1)-p(it1,1); y31=p(it3,2)-p(it1,2);
x32=p(it3,1)-p(it2,1); y32=p(it3,2)-p(it2,2);
ar=(x21.*y31-y21.*x31)/2;
if (nargout==1), return, end
a1=sqrt(x21.^2+y21.^2); a2=sqrt(x31.^2+y31.^2); a3=sqrt(x32.^2+y32.^2);
qt=(a2+a3-a1).*(a3+a1-a2).*(a1+a2-a3)./(a1.*a2.*a3);
if (nargout == 2) , return, end
te=[a1,a2,a3];
end