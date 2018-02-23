function [ ] = test_SimulatedAnnealingFit( )
%TEST_SIMULATEDANNEALINGFIT Test simulated annealing algorithm

lb      =   [-1,-1,-1];
ub      =   [ 1, 1, 1];
p0      =   2*rand(1,3)-1;

Results =   SimulatedAnnealingFit( @datafun, p0, lb, ub );

end

function f = datafun(params)

[x,y,z]         =   dealArray(params);

gaussian_fun	=	@(x,y,z,x0,y0,z0,s) exp(-((x-x0).^2+(y-y0).^2+(z-z0).^2)./(2*s^2));
[xl,yl,zl]      =	meshgrid(linspacePeriodic(-1,1,5));
[xl,yl,zl]      =   deal(xl(:),yl(:),zl(:));
peaks           =   1 + 0.3 * sin(linspace(-pi,pi,numel(xl)));

f	=	0;
s	=   0.05;
for ii = 1:numel(xl)
    f	=   f + peaks(ii) * gaussian_fun(x,y,z,xl(ii),yl(ii),zl(ii),s);
end

f	=	max(peaks(:)) - f;

end