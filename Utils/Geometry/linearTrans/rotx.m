function [ R ] = rotx( th )
%[ R ] = rotx( th ) Rotate by theta degrees about the x-axis

sin_th = sind(th);
cos_th = cosd(th);

R = [ 1, 0,       0
      0, cos_th, -sin_th
      0, sin_th,  cos_th ];

end

