function [q,v,d] = line3DFit(x,y,z)

% x,y,z are n x 1 column vectors of the three coordinates
% of a set of n points in three dimensions. The best line,
% in the minimum mean square orthogonal distance sense,
% will pass through q and have direction cosines in v, so
% it can be expressed parametrically as x = q(1) + v(1)*t,
% y = q(2) + v(2)*t, and z = q(3)+v(3)*t, where t is the
% distance along the line from the mean point at q.
% d returns with the minimum mean square orthogonal
% distance to the line.
% RAS - March 14, 2005

x=x(:);
y=y(:);
z=z(:);
n=size(x,1);
if ~all(n==[length(y) length(z)]);
    error('The arguments must be vectors of the same length.')
end

q = [mean(x),mean(y),mean(z)];
w = [x-q(1),y-q(2),z-q(3)]; % Use "mean" point as base
a = (1/n)*(w')*w; % 'a' is a positive definite matrix
[u,d,~] = svd(a); % 'eig' & 'svd' get same eigenvalues for this matrix
v = u(:,1)'; % Get eigenvector for largest eigenvalue
d = d(2,2)+d(3,3); % Sum the other two eigenvalues