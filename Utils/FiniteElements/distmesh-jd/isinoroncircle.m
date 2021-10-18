function b = isinoroncircle(p, center, radius, thresh)
%isinoroncircle(p, center, radius, thresh)
% Returns logical array of length size(p,1), indicating whether the
% corresponding point p(i,:) is within the circle to within `thresh`

% threshold means that you can be within `radius + thresh` from the origin
if nargin < 4; thresh = 0.0; end
radius = radius + thresh;
b = ((p(:,1) - center(1,1)).^2 + (p(:,2) - center(1,2)).^2 <= radius^2);

end
