function [ vx, vy, vz ] = nullVectors3D( vz )
%NULLVECTORS3D Gets the set of basis vectors vx and vy associated with vz
%such that V = [vx,vy,vz] is a proper rotation matrix, i.e. det(V) = 1 and
%V*V'=eye(3) to within working precision
% 
% INPUT ARGUMENTS
%   vz:	[3xN]	Vectors to find null-vectors for (need not be unit vectors)
% 
% OUTPUT ARGUMENTS
%   vx:	[3xN]   First principle null-vector
%   vy:	[3xN]	Second principle null-vector
%   vz: [3xN]   Same as vz, but normalized

vz      =   unit(vz,1);
[~,idx]	=   min(abs(vz),[],1);

xidx	=   (idx == 1);	nx	=   sum(xidx);
yidx	=   (idx == 2);	ny	=   sum(yidx);
zidx	=   (idx == 3);	nz	=   sum(zidx);
clear idx

% Let vz = [a;b;c]; any vector w = [x;y;z] normal to v satisfies w'*v = 0:
%
%   a*x + b*y + c*z = 0
% 
% Three solutions are:
%   wx	=	[ 0;-c; b ];
%   wy	=	[ c; 0;-a ];
%   wz	=   [-b; a; 0 ];
% 
% The solution that is chosen is the solution that does not use the
% smallest element of vz

vx	=	zeros(size(vz),'like',vz);

if nx > 0
    vx(:,xidx)	=   [ zeros([1,nx],'like',vz); -vz(3,xidx); vz(2,xidx) ];
end
if ny > 0
    vx(:,yidx)	=   [ vz(3,yidx); zeros([1,ny],'like',vz); -vz(1,yidx) ];
end
if nz > 0
    vx(:,zidx)	=   [ -vz(2,zidx); vz(1,zidx); zeros([1,nz],'like',vz) ];
end

vx	=   unit(vx,1);

% Take cross product to get vy = cross(vz,vx)
vy	=	[	vx(3,:).*vz(2,:) - vx(2,:).*vz(3,:)
            vx(1,:).*vz(3,:) - vx(3,:).*vz(1,:)
            vx(2,:).*vz(1,:) - vx(1,:).*vz(2,:)	];
vy	=   unit(vy,1);


end
