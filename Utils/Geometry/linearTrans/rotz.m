function [ R ] = rotz( th )
%[ R ] = rotz( th ) Rotate by theta degrees about the z-axis

sin_th = sind(th);
cos_th = cosd(th);

R = [ cos_th, -sin_th, 0
      sin_th,  cos_th, 0
      0,       0,      1 ];

end

