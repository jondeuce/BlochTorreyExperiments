function [ P ] = randOOBB( L, O, isAABB )
%RANDOOBB Returns a randomly oriented OOBB with dimensions L, a scalar or
%[3x1] vector (default [1,1,1]), and origin O (default [0,0,0]).
% 
%                  8 ________________ 7
%                   /|              /|
%                  / |             / |
%                 /  |            /  |
%                /   |           /   |
%               /    |          /    |
%              /   4 |_________/_____| 3
%             /     /         /     /
%          5 /_____/_________/ 6   /
%            |    /          |    /
%            |   /           |   /
%            |  /            |  /
%            | /             | /
%            |/______________|/
%          1                   2

if nargin < 2,      O = [0;0;0]; end
if nargin < 1,      L = [1;1;1];
elseif isscalar(L), L = [L;L;L]; end

if isAABB,	R	=	eye(3);
else        R	=	randRotMat;
end

[z,v12,v14,v15]	=   deal( [0;0;0], L(1)*R(:,1), L(2)*R(:,2), L(3)*R(:,3) );

P	=   bsxfun( @plus, O(:), ...
    [ z, v12, v12 + v14, v14, v15, v15 + v12, v15 + v12 + v14, v15 + v14 ] ...
    );

end

