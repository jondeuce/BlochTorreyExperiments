function h=spherePlot(r0,r,fignum,n)

if nargin < 4; n=20; end

theta = (-n:2:n)/n*pi;
phi = (-n:2:n)'/n*pi/2;
cosphi = cos(phi); cosphi(1) = 0; cosphi(n+1) = 0;
sintheta = sin(theta); sintheta(1) = 0; sintheta(n+1) = 0;

x = r*cosphi*cos(theta)+r0(1);
y = r*cosphi*sintheta+r0(2);
z = r*sin(phi)*ones(1,n+1)+r0(3);

figure(fignum)
hold on
hh=mesh(x,y,z);
hidden off
hold off

if nargout>0; h=hh; end;

end