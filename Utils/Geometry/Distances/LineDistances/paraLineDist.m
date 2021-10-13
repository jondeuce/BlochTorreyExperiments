function [D,D2] = paraLineDist( p0, q0, u )
%PARALINEDIST Finds the shortest distance between the parallel line pairs
%given by the line equations:
%       L1(:,i) = @(t) p0(:,i) + t * u(:,i)
%       L2(:,i) = @(t) q0(:,i) + s * u(:,i)

if any( [size(p0,1), size(q0,1), size(u,1)] ~= 3 )
    error('ERROR: all entries must have exactly 3 rows');
end

%% fast execution

% Hardcoded function call for speed (see below for clear derivation):
% 
% D2 = f_D2( p0(1), q0(1), p0(2), q0(2), p0(3), q0(3), u(1), u(2), u(3) );
% 
% where:
% 
% f_D2 = @(a0,a1,b0,b1,c0,c1,x,y,z) ...
%     (-a0+a1+(x.*(a0.*x-a1.*x+b0.*y-b1.*y+c0.*z-c1.*z))./(x.^2+y.^2+z.^2)).^2 + ...
%     (-b0+b1+(y.*(a0.*x-a1.*x+b0.*y-b1.*y+c0.*z-c1.*z))./(x.^2+y.^2+z.^2)).^2 + ...
%     (-c0+c1+(z.*(a0.*x-a1.*x+b0.*y-b1.*y+c0.*z-c1.*z))./(x.^2+y.^2+z.^2)).^2;

num	=   (p0(1,:).*u(1,:)-q0(1,:).*u(1,:)+p0(2,:).*u(2,:)-q0(2,:).*u(2,:)+p0(3,:).*u(3,:)-q0(3,:).*u(3,:));
den	=   (u(1,:).^2+u(2,:).^2+u(3,:).^2);
D	=	(-p0(1,:)+q0(1,:)+(u(1,:).*num)./den).^2 + ...
        (-p0(2,:)+q0(2,:)+(u(2,:).*num)./den).^2 + ...
        (-p0(3,:)+q0(3,:)+(u(3,:).*num)./den).^2;
clear num den

if nargout > 1
    D2 = D;
end
D	=	sqrt(D);

%% parallel line derivation
% syms a0 b0 c0 a1 b1 c1 x y z s t real
% 
% p0 = [a0,b0,c0]';
% q0 = [a1,b1,c1]';
% u = [x,y,z]';
% 
% Lp = p0 + t*u;
% Lq = q0 + s*u;
% 
% d2 = sum( (Lp - Lq).^2 );
% f_d2 = matlabFunction(d2);
% 
% sol = solve( [ diff(d2,t); diff(d2,s) ], t, s );
% 
% T = sol.t;
% S = sol.s;
% 
% D2 = simplify( f_d2(a0,a1,b0,b1,c0,c1,S,T,x,y,z) );
% f_D2 = matlabFunction(D2);
% 
% result:
% f_D2 = @(a0,a1,b0,b1,c0,c1,x,y,z) ...
%     (-a0+a1+(x.*(a0.*x-a1.*x+b0.*y-b1.*y+c0.*z-c1.*z))./(x.^2+y.^2+z.^2)).^2 + ...
%     (-b0+b1+(y.*(a0.*x-a1.*x+b0.*y-b1.*y+c0.*z-c1.*z))./(x.^2+y.^2+z.^2)).^2 + ...
%     (-c0+c1+(z.*(a0.*x-a1.*x+b0.*y-b1.*y+c0.*z-c1.*z))./(x.^2+y.^2+z.^2)).^2;

%% test
%{
N  = 10000;
p0 = 2*rand(3,N)-1;
q0 = 2*rand(3,N)-1;
u  = [ones(1,N);  zeros(1,N); zeros(1,N)]; %x-direction
% u  = [zeros(1,N); ones(1,N);  zeros(1,N)]; %y-direction
% u  = [zeros(1,N); zeros(1,N); ones(1,N) ]; %z-direction

% parallel lines of known distance
d1 = sqrt( sum( ( p0([2,3],:) - q0([2,3],:) ).^2, 1 ) ); %x-direction
% d1 = sqrt( sum( ( p0([1,3],:) - q0([1,3],:) ).^2, 1 ) ); %y-direction
% d1 = sqrt( sum( ( p0([1,2],:) - q0([1,2],:) ).^2, 1 ) ); %z-direction
[d2,D2] = paraLineDist( p0, q0, u );

tol = eps(max(max(d1(:)),max(d2(:))));
if N == 1
    disp( [d1, d2, abs(d1-d2) < tol] );
else
    disp( all( abs(d1-d2) < tol ) );
end
%}

end

