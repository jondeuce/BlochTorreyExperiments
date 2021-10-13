function phi = angle_between_points( x1, y1, x2, y2, direction )
%ANGLE_BETWEEN_POINTS calculates the angle between the pairs of points
%P1 = (x1,y1) and P2 = (x2,y2) from P1 to P2 in the direction 'direction'
%   -direction == 0 is the shortest angle (default)
%   -direction == 1 is the CCW angle
%   -direction == 2 is the CW angle

if nargin < 5; direction = 1; end

% shortest angle between vectors in 2D, based on the more general 3D version
%   phi = atan2(norm(cross(vec1(:),vec2(:))),dot(vec1(:),vec2(:)));
cross_prod = x1.*y2 - x2.*y1;
dot_prod = x1.*x2 + y1.*y2;

if direction == 1
    %% angle from P1 CCW to P2
    phi = atan2(cross_prod,dot_prod); % phi is in (-pi,pi]
    phi(abs(phi)<100*eps) = 0; % set small angles to 0
    phi = phi + 2*pi*(phi<0); % phi is in [0,2*pi)
elseif direction == 2
    %% angle from P1 CW to P2
    phi = atan2(cross_prod,dot_prod); % phi is in (-pi,pi]
    phi(abs(phi)<100*eps) = 0; % set small angles to 0
    phi = 2*pi*(phi>0) - phi; % phi is in [0,2*pi)
else
    %% angle from P1 CW to P2
    phi = atan2(abs(cross_prod),dot_prod); % phi is in (-pi,pi]
    phi(abs(phi)<100*eps) = 0; % set small angles to 0
end

end

% function test_angle_between_points
% % run this in terminal for testing
% 
% size=[10,1];
% th1 = 2*pi*rand(size); th2 = 2*pi*rand(size); r1 = rand(size); r2 = rand(size);
% x1=r1.*cos(th1); y1=r1.*sin(th1); x2=r2.*cos(th2); y2=r2.*sin(th2);
% dth = mod(th2-th1,2*pi); dth(abs(dth)<100*eps) = 0; dth = dth + 2*pi*(dth<0);
% 
% disp([char(10) 'CCW: Random Angles']);
% disp('    Expected  Resulting');
% A = [dth, angle_between_points(x1,y1,x2,y2,1)];
% disp(A); if any(abs(diff(A,1,2))>100*eps); disp('FAILURE'); end
% 
% th1 = 2*pi*rand(2,1); th2 = [th1(1); th1(2)+4*pi]; r1 = rand(2,1); r2 = rand(2,1);
% x1=r1.*cos(th1); y1=r1.*sin(th1); x2=r2.*cos(th2); y2=r2.*sin(th2);
% dth = mod(th2-th1,2*pi); dth(abs(dth)<100*eps) = 0; dth = dth + 2*pi*(dth<0);
% 
% disp([char(10) 'CCW: Boundary Angles']);
% disp('    Expected  Resulting');
% A = [dth, angle_between_points(x1,y1,x2,y2,1)];
% disp(A); if any(abs(diff(A,1,2))>100*eps); disp('FAILURE'); end
% 
% 
% size=[10,1];
% th1 = 2*pi*rand(size); th2 = 2*pi*rand(size); r1 = rand(size); r2 = rand(size);
% x1=r1.*cos(th1); y1=r1.*sin(th1); x2=r2.*cos(th2); y2=r2.*sin(th2);
% dth = mod(th1-th2,2*pi); dth(abs(dth)<100*eps) = 0; dth = dth + 2*pi*(dth<0);
% 
% disp([char(10) 'CW: Random Angles']);
% disp('    Expected  Resulting');
% A = [dth, angle_between_points(x1,y1,x2,y2,2)];
% disp(A); if any(abs(diff(A,1,2))>100*eps); disp('FAILURE'); end
% 
% th1 = 2*pi*rand(2,1); th2 = [th1(1); th1(2)+4*pi]; r1 = rand(2,1); r2 = rand(2,1);
% x1=r1.*cos(th1); y1=r1.*sin(th1); x2=r2.*cos(th2); y2=r2.*sin(th2);
% dth = mod(th1-th2,2*pi); dth(abs(dth)<100*eps) = 0; dth = dth + 2*pi*(dth<0);
% 
% disp([char(10) 'CW: Boundary Angles']);
% disp('    Expected  Resulting');
% A = [dth, angle_between_points(x1,y1,x2,y2,2)];
% disp(A); if any(abs(diff(A,1,2))>100*eps); disp('FAILURE'); end
% 
% 
% size=[10,1];
% th1 = 2*pi*rand(size); th2 = 2*pi*rand(size); r1 = rand(size); r2 = rand(size);
% x1=r1.*cos(th1); y1=r1.*sin(th1); x2=r2.*cos(th2); y2=r2.*sin(th2);
% dth = min(abs(mod([th1-th2,th2-th1],2*pi)),[],2); dth(abs(dth)<100*eps) = 0;
% 
% disp([char(10) 'MIN: Random Angles']);
% disp('    Expected  Resulting');
% A = [dth, angle_between_points(x1,y1,x2,y2,0)];
% disp(A); if any(abs(diff(A,1,2))>100*eps); disp('FAILURE'); end
% 
% th1 = 2*pi*rand(4,1); th2 = [th1(1); th1(2)-pi; th1(3)+pi; th1(4)+2*pi]; r1 = rand(4,1); r2 = rand(4,1);
% x1=r1.*cos(th1); y1=r1.*sin(th1); x2=r2.*cos(th2); y2=r2.*sin(th2);
% dth = min(abs(mod([th1-th2,th2-th1],2*pi)),[],2); dth(abs(dth)<100*eps) = 0;
% 
% disp([char(10) 'MIN: Boundary Angles']);
% disp('    Expected  Resulting');
% A = [dth, angle_between_points(x1,y1,x2,y2,0)];
% disp(A); if any(abs(diff(A,1,2))>100*eps); disp('FAILURE'); end
% 
% end