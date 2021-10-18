function [p, v, r] = sampleRandomCylinders( BoxDims, BoxCenter, r, N, type )
%SAMPLERANDOMCYLINDERS Summary of this function goes here
%   Detailed explanation goes here

BoxDims = BoxDims(:);
BoxCenter = BoxCenter(:);

if nargin < 5; type = 'BoundingSphereAndDirection'; end

switch upper(type)
    case 'POINTANDDIRECTION'
        [p, v] = randomPointRandomDirection( BoxDims, BoxCenter, r, N );
    case 'BOUNDINGSPHEREPAIR'
        [p, v] = randomLineThroughBoundingSphere( BoxDims, BoxCenter, r, N );
    otherwise % 'BOUNDINGSPHEREANDDIRECTION'
        [p, v] = randomPointOnBoundingSphereRandomDirection( BoxDims, BoxCenter, r, N );
end

end

function v = randomUnitSpherePoints(N)
% helper function generating N random points on the unit sphere
v = normalizePoints(randn(3,N));
end

function v = normalizePoints(v)
% normalize unit vectors
v = bsxfun(@rdivide, v, sqrt(sum(v.^2,1)));
end

function [p, v] = randomPointRandomDirection( BoxDims, BoxCenter, r, N )
% This function generates uniformly random unit vectors v, and uniformly
% random points p within the box defined by BoxDims and BoxCenter

v = randomUnitSpherePoints(N);
v(3,:) = abs(v(3,:)); % force positive z-component to avoid degeneracy
p = bsxfun( @plus, BoxCenter, bsxfun( @times, BoxDims/2, 2*rand(3,N)-1 ) );

% if N == 1
%     p = bsxfun( @plus, BoxCenter, ...
%                             bsxfun( @times, BoxDims/2-r, ...
%                                             2*rand(3,N)-1 ) );
% else
%     p = bsxfun( @plus, BoxCenter, ...
%                        bsxfun( @times, bsxfun( @minus, BoxDims/2, r ), ...
%                                             2*rand(3,N)-1 ) );
% end

end

function [p, v] = randomLineThroughBoundingSphere( BoxDims, BoxCenter, r, N )
% This function generates a more uniform distributions of cylinders.
% Essentially, consider pairs of uniformly random points on the sphere
% bounding the box defined by BoxDims and BoxCenter. The line through
% these points is taken to be the cylinder axis. This produces more
% uniformly random cylinders in the sense that a histogram of the number
% of points within cylinders corresponding to a certain angle is more
% uniform.

[p, v] = NrandomLinesThroughBoundingSphere( BoxDims, BoxCenter, r, N );
while size(p,2) < N
    [p_, v_] = NrandomLinesThroughBoundingSphere( BoxDims, BoxCenter, r, N );
    p = [p, p_];
    v = [v, v_];
end
p = p(:, 1:N);
v = v(:, 1:N);

end

function [p, v] = NrandomLinesThroughBoundingSphere( BoxDims, BoxCenter, r, N )
% Helper function for above

R = norm(BoxDims)/2; % half-length of box diagonal/radius of bounding sphere
translateAndScale = @(v) bsxfun(@plus, R * v, BoxCenter);
p1 = translateAndScale(randomUnitSpherePoints(N));
p2 = translateAndScale(randomUnitSpherePoints(N));
p = (p1+p2)/2; % cylinder axis point
v = normalizePoints(p1-p2);
v(3,:) = abs(v(3,:)); % cylinder direction redundancy

% tmin is the location of the first intersection point with the box. If it
% is NaN, there is no intersection
tmin = rayBoxIntersection( p, v, BoxDims, BoxCenter );
b = ~isnan(tmin);
p = p(:, b);
v = v(:, b);

end

function [p, v] = randomPointOnBoundingSphereRandomDirection( BoxDims, BoxCenter, r, N )
% This function is similar to the above, except now the second random point
% on the on the unitsphere is interpreted as the direction of the cylinder

[p, v] = NrandomPointsOnBoundingSphereRandomDirection( BoxDims, BoxCenter, r, N );
while size(p,2) < N
    [p_, v_] = NrandomPointsOnBoundingSphereRandomDirection( BoxDims, BoxCenter, r, N );
    p = [p, p_];
    v = [v, v_];
end
p = p(:, 1:N);
v = v(:, 1:N);

end

function [p, v] = NrandomPointsOnBoundingSphereRandomDirection( BoxDims, BoxCenter, r, N )
% Helper function for above

R = norm(BoxDims)/2; % half-length of box diagonal/radius of bounding sphere
translateAndScale = @(v) bsxfun(@plus, R * v, BoxCenter);
p = translateAndScale(randomUnitSpherePoints(N)); % cylinder axis point
v = randomUnitSpherePoints(N); % cylinder axis direction
v(3,:) = abs(v(3,:));

% tmin is the location of the first intersection point with the box. If it
% is NaN, there is no intersection. Lmid is the point on the cylinder in
% the middle, between the two intersection points.
[tmin, ~, ~, ~, Lmid] = rayBoxIntersection( p, v, BoxDims, BoxCenter );
b = ~isnan(tmin);
p = Lmid(:, b);
v = v(:, b);

end