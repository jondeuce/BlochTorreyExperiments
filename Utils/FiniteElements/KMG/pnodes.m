function [pp,iter]=pnodes(p,fd,h,tol,varargin)
%PNODES Project external nodes onto the boundary
%
%
d=feval(fd,p,varargin{:}); 
iter=0; pp=p;
while (max(abs(d))>tol & iter<10) 
    ddx=(feval(fd,[pp(:,1)+h,pp(:,2)],varargin{:})-d)/h;
    ddy=(feval(fd,[pp(:,1),pp(:,2)+h],varargin{:})-d)/h;
    pp=pp-[d.*ddx,d.*ddy];
    d=feval(fd,pp,varargin{:});
    iter=iter+1;
end

