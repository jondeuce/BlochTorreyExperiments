function test_eulerDerivative
%TEST_EULERDERIVATIVE Summary of this function goes here
%   Detailed explanation goes here

Ts = 1/125;
w_max = 10;

% w = w_max * ( 2*rand(3,1)-1 );
w = [0.5;1;2];
W = norm(w);

% R = randRotMat;
% R = eye(3);
% [psi0,th0,phi0] = attitude2Euler(R);
phi0 =  1.2;
th0  =  0.2;
psi0 = -0.8;

numIter = 10;
[psi, th, phi] = deal( zeros(numIter, 1) );
psi(1) = psi0;
th(1)  = th0;
phi(1) = phi0;
for n = 1:numIter-1
    dTh = eulerDerivative(th(n),phi(n),w);
    psi(n+1) = psi0 + Ts * dTh(3);
    th(n+1)  = th0  + Ts * dTh(2);
    phi(n+1) = phi0 + Ts * dTh(1);
end

disp( [ phi, th, psi ] );

% J = @(phi,th,psi,p,q,r) ...
%     [     q*cos(phi)*tan(th) - r*sin(phi)*tan(th),         r*cos(phi)*(tan(th)^2 + 1) + q*sin(phi)*(tan(th)^2 + 1), 0
%                         - r*cos(phi) - q*sin(phi),                                                               0, 0
%       (q*cos(phi))/cos(th) - (r*sin(phi))/cos(th), (r*cos(phi)*sin(th))/cos(th)^2 + (q*sin(phi)*sin(th))/cos(th)^2, 0 ];
% 
% Th  = [phi0; th0; psi0];
% Th0 = [0;0;0];
% h   = Ts/(W/5);
% while norm(Th-Th0)>1e-3
%     Th0 = Th;
%     Th  = Th0 - h * J(Th0(1), Th0(2), Th0(3), w(1), w(2), w(3)) * w;
% end
% 
% disp( Th' );

end

