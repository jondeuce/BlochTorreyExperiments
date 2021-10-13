function [x,y,z] = rotationDecompose( R )
%ROTATIONDECOMPOSE decomposes the right rotation matrix R into euler angles
%x,y,z such that the corresponding rotation matrices X,Y,Z (which act to
%rotate about the X,Y,Z axes) are the factors of R.
% 
%   i.e. R = X*Y*Z

%% Demo for random composite rotation matrices
% rotation_matrix_demo

%% decompose the rotation matrix R
[x,y,z]=decompose_rotation(R);

end

function [x,y,z] = decompose_rotation(R)
	x = atan2(R(2,3), R(3,3));
	y = atan2(-R(1,3), sqrt(R(2,3)*R(2,3) + R(3,3)*R(3,3)));
	z = atan2(R(1,2), R(1,1));
end

function rotation_matrix_demo

	disp('Picking random Euler angles (radians)');

	x = 2*pi*(rand-0.5); % -180 to 180
	y = pi*(rand-0.5); % -90 to 90
	z = 2*pi*(rand-0.5); % -180 to 180
    disp([x;y;z]);
    
	disp('Rotation matrix is:');
	R = compose_rotation(x,y,z);
    disp(R);
	
	disp('Decomposing R:');
	[x2,y2,z2] = decompose_rotation(R);
    disp([x2;y2;z2]);

    disp('Error:');
	err = sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y) + (z2-z)*(z2-z));
    disp(err);

	if err < 1e-5
		disp('Results are correct!');
	else
		disp('Results are incorrect.');
	end
end

function R = compose_rotation(x,y,z)

Rx=@(x)[1,0,0;0,cos(x),sin(x);0,-sin(x),cos(x)];
Ry=@(y)[cos(y),0,-sin(y);0,1,0;sin(y),0,cos(y)];
Rz=@(z)[cos(z),sin(z),0;-sin(z),cos(z),0;0,0,1];

R=Rx(x)*Ry(y)*Rz(z);

end