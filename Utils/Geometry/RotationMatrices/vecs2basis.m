function N = vecs2basis( x0, y0, x1, y1 )
%ROTMAT_2_VECS Summary of this function goes here
%   Detailed explanation goes here

dth = angle_3D_rads(x0(:),y0(:)) - angle_3D_rads(x1(:),y1(:));
R = Ru(unit(cross(x1(:),y1(:))),dth/2);
x1 = R'*x1(:);
y1 = R*y1(:);

N = [unit(x0(:)), unit(y0(:)), unit(cross(x0(:),y0(:)))] / ...
    [unit(x1(:)), unit(y1(:)), unit(cross(x1(:),y1(:)))];

end

function x = unit(x)
x = x(:)/norm(x);
end

function theta = angle_3D_rads(vec1,vec2)
theta = atan2(norm(cross(vec1(:),vec2(:))),dot(vec1(:),vec2(:)));
end

function Rot = Ru(u,t)
% arbitrary rotation matrix
Rot = cos(t)*eye(3) + (1-cos(t))*kron(u,u') + sin(t)*skew(u);
end

function A = skew(u)
% skew symmetric matrix
A = [ 0,-u(3),u(2); u(3),0,-u(1); -u(2),u(1),0 ];
end