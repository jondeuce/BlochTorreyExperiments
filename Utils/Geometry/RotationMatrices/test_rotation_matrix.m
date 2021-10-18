function R = test_rotation_matrix
%TEST_ROTATION_MATRIX Summary of this function goes here
%   Detailed explanation goes here

w = [0.1; 0.05; -0.2];
omega = norm(w);
dt = 1;

R = axisAngle2Rot(w/omega, omega*dt);

gravity = 9.81;
g = [0;0;gravity];
g = R*g;

figure, hold on, grid on
axis([-1,1,-1,1,-1,1]);

org = [0,0,0];
hx = arrow(org, R(1,:)', 'facecolor', 'r');
hy = arrow(org, R(2,:)', 'facecolor', 'g');
hz = arrow(org, R(3,:)', 'facecolor', 'b');

gx = arrow(org, R(1,:)', 'facecolor', 'r');
gy = arrow(org, R(2,:)', 'facecolor', 'g');
gz = arrow(org, R(3,:)', 'facecolor', 'b');


legend([hx;hy;hz],'x','y','z');

axis image

end