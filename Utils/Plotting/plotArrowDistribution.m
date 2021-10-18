function plotArrowDistribution( thetaMin, thetaMax, Ntotal, facecolor )
%PLOTARROWDISTRIBUTION Plots distribution of arrows on sphere with angle
%thetaMin <= theta <= thetaMax, where theta is the polar angle

if nargin < 4; facecolor = 'b'; end
if nargin < 3; Ntotal = 1000; end
if nargin < 2; thetaMax = 10.0; end
if nargin < 1; thetaMin = 0.0; end

% Initialize plot
close all force
figure, hold on
axis off
axis image
axis([-1,1,-1,1,-1,1])
view([1,1,0.6])
% view([0,0,1])

% Plot sphere
[x,y,z] = sphere;
surf(x,y,z,'facealpha',0.01,'linestyle',':','facecolor','k')

% Plot arrows
[x,y,z,Th,Phi] = uniformSphere(Ntotal);

Th = 180/pi*Th; % degrees
Th = min(Th, 180-Th); % unsigned angle w.r.t vertical

Ntotal = size(x,1); % Ntotal actually simulated may be less
idx = (thetaMin <= Th & Th <= thetaMax);
[x,y,z,th,phi] = deal(x(idx),y(idx),z(idx),Th(idx),Phi(idx));

Nfrac = sum(idx);
frac = Nfrac/Ntotal;
disp(frac)

% plot3(x,y,z,'bo')
starts = zeros(Nfrac,3);
stops = [x,y,z];
NormalDir = [cos(th).*cos(phi), cos(th).*sin(phi), -sin(th)];
CrossDir = [-sin(phi), cos(phi), zeros(size(phi))];
axis(axis);
arrow( ...
    'start',     starts, ...
    'stop',      stops, ...
    'NormalDir', NormalDir, ...
    'CrossDir',  CrossDir, ...
    'length',    20, ...
    'tipangle',  60, ...
    'baseangle', 50, ...
    'tipangle',  [], ...
    'width',     3, ...
    'EdgeColor', 'k', ...
    'FaceColor', facecolor, ...
    'LineWidth', 1 ...
    );

end

function [x,y,z,Th,Phi] = uniformSphere(N,r)

if nargin < 2; r = 1; end
if nargin < 1; N = 100; end

Nc = 1;
a = 4*pi*r^2/N;
d = sqrt(a);

Mth = round(pi/d);
dth = pi/Mth;
dphi = a/dth;

[x,y,z,Th,Phi] = deal(zeros(N,1));

for m = 1:Mth
    %For each m in 0 . . .Mth ? 1 do {
    th = pi*(m - 0.5)/Mth;
    Mphi = round(2*pi*sin(th)/dphi);
    for n = 1:Mphi
        % For each n in 0 . . .Mphi ? 1 do {
        phi = 2*pi*(n-1)/Mphi;
        % Create point using Eqn. (1).
        x(Nc) = r*sin(th)*cos(phi);
        y(Nc) = r*sin(th)*sin(phi);
        z(Nc) = r*cos(th);
        Th(Nc) = th;
        Phi(Nc) = phi;
        Nc = Nc + 1;
    end
end

x = x(1:Nc-1);
y = y(1:Nc-1);
z = z(1:Nc-1);
Th = Th(1:Nc-1);
Phi = Phi(1:Nc-1);

end

function th = angle3D_degrees(vec1,vec2,dim)

if nargin < 3; dim = find(size(vec1)==3); end

th = 180/pi * atan2( norm( cross( vec1, vec2, dim ) ), ...
                             dot( vec1, vec2, dim ) );

end