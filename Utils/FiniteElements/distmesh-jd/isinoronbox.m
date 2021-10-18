function b = isinoronbox(p, bbox, thresh)
%ISINORONBOX Returns logical array of length size(p,1), indicating whether
%the corresponding point p(i,:) is strictly within the box bbox.

if nargin < 3
    b = bbox(1,1) <= p(:,1) & p(:,1) <= bbox(2,1) & ...
        bbox(1,2) <= p(:,2) & p(:,2) <= bbox(2,2);
else
    b = bbox(1,1) - thresh <= p(:,1) & p(:,1) <= bbox(2,1) + thresh & ...
        bbox(1,2) - thresh <= p(:,2) & p(:,2) <= bbox(2,2) + thresh;
end

end