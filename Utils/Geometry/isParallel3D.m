function [ b ] = isParallel3D( v1, v2, tol )
%ISPARALLEL3D Returns true if the vectors v1 and v2 are parallel to within
%tol. v1 and v2 must each be either [3xN] or [3x1] for the same N.

if nargin < 3
    tol	=   10 * eps( max( max(abs(v1),[],1), max(abs(v2),[],1) ) );
end

v	=	[	v2(3,:).*v1(2,:) - v2(2,:).*v1(3,:)
            v2(1,:).*v1(3,:) - v2(3,:).*v1(1,:)
            v2(2,:).*v1(1,:) - v2(1,:).*v1(2,:)	];

b	=   max(abs(v),[],1) < tol;

end

