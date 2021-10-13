function [ b ] = iscircleboxintersect( bbox, centers, radii )
%ISCIRCLEBOXINTERSECT Checks if the box bbox intersects the circles defined
%by centers and radii

xmins = min(abs(centers(:,1) - bbox(1,1)), abs(centers(:,1) - bbox(2,1)));
ymins = min(abs(centers(:,2) - bbox(1,2)), abs(centers(:,2) - bbox(2,2)));
dmins = min(xmins, ymins);
b = dmins <= radii;

end
