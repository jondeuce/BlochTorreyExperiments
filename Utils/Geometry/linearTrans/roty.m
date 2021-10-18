function [ R ] = roty( th )
%[ R ] = roty( th ) Rotate by theta degrees about the y-axis

sin_th = sind(th);
cos_th = cosd(th);

R = [ cos_th, 0, sin_th
      0,      1, 0
     -sin_th, 0, cos_th];

end

