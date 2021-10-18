function circlePoints = addcircleboundarypoints(circlePoints, bbox, centers, radii)
%ADDCIRCLEBOUNDARYPOINTS(circlePoints, bbox, centers, radii)

b = iscircleboxintersect( bbox, centers, radii );
idx = find(b(:).');
for ii = idx
    % (x-a)^2 + (y-b)^2 = r^2  -->  y = +/-sqrt( r^2 - (x-a)^2 ) + b
    a = centers(ii,1);
    b = centers(ii,2);
    r = radii(ii);
    
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
    
    if ~isempty(points)
        circlePoints = [circlePoints; points];
    end
end

end