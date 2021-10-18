function circlePoints = gencirclepoints(h0, centers, radii, nminpoints)
%GENCIRCLEPOINTS Generate uniformly spaced points on the circles specified
%by `centers` and `radii`. The distance between adjacent points will be
%approximately h0. 

if nargin < 4; nminpoints = 8; end

Ns = round(2*pi*radii/h0);
Ns = max(Ns,nminpoints); %circle should have at least `nminpoints` points
Ntotal = sum(Ns);
circlePoints = zeros(Ntotal, 2);

idx = 1;
for ii = 1:length(radii)
    r = radii(ii);
    N = Ns(ii);
    th = linspacePeriodic(0,2*pi,N).';
    circlePoints(idx:idx+N-1,:) = bsxfun(@plus, r*[cos(th), sin(th)], centers(ii,:));
    idx = idx + N;
end

end