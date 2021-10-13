function circlePoints = gencirclepointsinbox(h0, bbox, centers, radii, nminpoints)
%GENCIRCLEPOINTS Generate uniformly spaced points on the circles specified
%by `centers` and `radii`. The distance between adjacent points will be
%approximately h0.

if nargin < 5; nminpoints = 8; end

Ns = round(2*pi*radii/h0);
Ns = max(Ns,nminpoints); %circle should have at least `nminpoints` points
Ntotal = sum(Ns);
circlePoints = zeros(Ntotal, 2);

idx = 1;
for ii = 1:length(radii)
    r = radii(ii);
    N = Ns(ii);

    bdry = possiblecircleboxintersections(bbox, centers(ii,:), r);
    if ~isempty(bdry)
        bdry = real(bdry(all(abs(imag(bdry)) < 10*eps, 2), :));
    end
    if ~isempty(bdry)
        bdry = bdry(isonbox(bdry, bbox, sqrt(eps)), :);
    end
    % bdry = circleBoundaryPoints(bbox, centers(ii,:), r);

    if isempty(bdry)
        th = linspacePeriodic(-pi,pi,N).';
    else
        th_bdry = unique(atan2(bdry(:,2)-centers(ii,2), bdry(:,1)-centers(ii,1)));

        if numel(th_bdry) == 1
            th = linspacePeriodic(0,2*pi,N).';
            th = atan2range(th-th(1)+th_bdry);
        else
            th = [];
            th_bdry_padded = [th_bdry; th_bdry(1)+2*pi];
            dths = circshift(th_bdry_padded,-1) - th_bdry_padded;
            for jj = 1:length(th_bdry)
                Nth = max(round(N*dths(jj)/(2*pi)), 2);
                th_chunk = linspace(th_bdry(jj),th_bdry_padded(jj+1),Nth).';
                th = [th; th_chunk(1:end-1)];
            end
            th = atan2range(th);
        end
    end

    N = length(th);
    Ns(ii) = N;
    Ntotal = sum(Ns);

    circlePoints(idx:idx+N-1,:) = bsxfun(@plus, r*[cos(th), sin(th)], centers(ii,:));
    idx = idx + N;
end

circlePoints = circlePoints(1:Ntotal, :);

end

function th = atan2range(th) % shift to [-pi,pi]
th = mod(th+pi,2*pi)-pi;
end

function points = circleBoundaryPoints(bbox, center, radius)

if iscircleboxintersect( bbox, center, radius )
    % (x-a)^2 + (y-b)^2 = r^2  -->  y = +/-sqrt( r^2 - (x-a)^2 ) + b
    a = center(1,1);
    b = center(1,2);
    r = radius;

    points = zeros(8,2);
    cnt = 0;

    x = bbox(1,1);
    if x <= a && a-x <= r
        d = sqrt( r^2 - (x-a)^2 );
        points(cnt+1, :) = [x,  d + b];
        points(cnt+2, :) = [x, -d + b];
        cnt = cnt + 2;
    end

    x = bbox(2,1);
    if a <= x && x-a <= r
        d = sqrt( r^2 - (x-a)^2 );
        points(cnt+1, :) = [x,  d + b];
        points(cnt+2, :) = [x, -d + b];
        cnt = cnt + 2;
    end

    y = bbox(1,2);
    if y <= b && b-y <= r
        d = sqrt( r^2 - (y-b)^2 );
        points(cnt+1, :) = [ d + a, y];
        points(cnt+2, :) = [-d + a, y];
        cnt = cnt + 2;
    end

    y = bbox(2,2);
    if b <= y && y-b <= r
        d = sqrt( r^2 - (y-b)^2 );
        points(cnt+1, :) = [ d + a, y];
        points(cnt+2, :) = [-d + a, y];
        cnt = cnt + 2;
    end

    points = points(1:cnt, :);
    points = points(isinoronbox(points, bbox), :);
else
    points = zeros(0,2);
end

end
