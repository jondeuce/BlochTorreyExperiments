function [D,D2] = skewLineDist( p0, q0, p, q )
%SKEWLINEDIST Finds the shortest distance between the skew line pairs given
%by the line equations:
%       L1(:,i) = @(t) p0(:,i) + t * p(:,i)
%       L2(:,i) = @(s) q0(:,i) + s * q(:,i)

if any( [size(p0,1), size(q0,1), size(p,1), size(q,1)] ~= 3 )
    error('ERROR: all entries must have exactly 3 rows');
end

%% fast execution

% Hardcoded function call for speed (see below for clear derivation):
% 
% D2 = f_D2( p0(1), q0(1), p0(2), q0(2), p0(3), q0(3), ...
%            p(1),  q(1),  p(2),  q(2),  p(3),  q(3) );
% 
% where:
% 
% f_D2 = @(a0,a1,b0,b1,c0,c1,x0,x1,y0,y1,z0,z1) ...
%     (a0.*y0.*z1-a0.*y1.*z0-b0.*x0.*z1+b0.*x1.*z0+c0.*x0.*y1-c0.*x1.*y0-a1.*y0.*z1+a1.*y1.*z0+b1.*x0.*z1-b1.*x1.*z0-c1.*x0.*y1+c1.*x1.*y0).^2 ./ ...
%     (x0.^2.*y1.^2+x1.^2.*y0.^2+x0.^2.*z1.^2+x1.^2.*z0.^2+y0.^2.*z1.^2+y1.^2.*z0.^2-x0.*x1.*y0.*y1.*2.0-x0.*x1.*z0.*z1.*2.0-y0.*y1.*z0.*z1.*2.0);

D2_denominator = ...
        (   p(1,:).^2.*q(2,:).^2+q(1,:).^2.*p(2,:).^2 + ...
            p(1,:).^2.*q(3,:).^2+q(1,:).^2.*p(3,:).^2 + ...
            p(2,:).^2.*q(3,:).^2+q(2,:).^2.*p(3,:).^2 - ...
            p(1,:).*q(1,:).*p(2,:).*q(2,:).*2.0 - ...
            p(1,:).*q(1,:).*p(3,:).*q(3,:).*2.0 - ...
            p(2,:).*q(2,:).*p(3,:).*q(3,:).*2.0     );

D	=	(   p0(1,:).*p(2,:).*q(3,:)-p0(1,:).*q(2,:).*p(3,:) -   ...
            p0(2,:).*p(1,:).*q(3,:)+p0(2,:).*q(1,:).*p(3,:) +   ...
            p0(3,:).*p(1,:).*q(2,:)-p0(3,:).*q(1,:).*p(2,:) -   ...
            q0(1,:).*p(2,:).*q(3,:)+q0(1,:).*q(2,:).*p(3,:) +   ...
            q0(2,:).*p(1,:).*q(3,:)-q0(2,:).*q(1,:).*p(3,:) -   ...
            q0(3,:).*p(1,:).*q(2,:)+q0(3,:).*q(1,:).*p(2,:)	).^2	./	...
            D2_denominator;

if nargout > 1
    D2	=	D;
end
D	=	sqrt(D);

tol	=   5*eps(class(D2_denominator));
ix	=	abs( D2_denominator ) < tol;
if any( ix );
    if nargout > 1
        [D(ix),D2(ix)]	=	paraLineDist( p0(:,ix), q0(:,ix), (p(:,ix) + q(:,ix))/2 );
    else
        D(ix)           =	paraLineDist( p0(:,ix), q0(:,ix), (p(:,ix) + q(:,ix))/2 );
    end
end

%% skew line derivation
% syms a0 b0 c0 a1 b1 c1 x0 y0 z0 x1 y1 z1 s t real
% 
% p0 = [a0,b0,c0]';
% q0 = [a1,b1,c1]';
% p = [x0,y0,z0]';
% q = [x1,y1,z1]';
% 
% Lp = p0 + t*p;
% Lq = q0 + s*q;
% 
% d2 = sum( (Lp - Lq).^2 );
% f_d2 = matlabFunction(d2);
% 
% sol = solve( [ diff(d2,t); diff(d2,s) ], t, s );
% 
% T = sol.t;
% S = sol.s;
% 
% D2 = simplify( f_d2(a0,a1,b0,b1,c0,c1,S,T,x0,x1,y0,y1,z0,z1) );
% f_D2 = matlabFunction(D2);
% 
% % result:
% f_D2 = @(a0,a1,b0,b1,c0,c1,x0,x1,y0,y1,z0,z1) ...
%     (a0.*y0.*z1-a0.*y1.*z0-b0.*x0.*z1+b0.*x1.*z0+c0.*x0.*y1-c0.*x1.*y0-a1.*y0.*z1+a1.*y1.*z0+b1.*x0.*z1-b1.*x1.*z0-c1.*x0.*y1+c1.*x1.*y0).^2 ./ ...
%     (x0.^2.*y1.^2+x1.^2.*y0.^2+x0.^2.*z1.^2+x1.^2.*z0.^2+y0.^2.*z1.^2+y1.^2.*z0.^2-x0.*x1.*y0.*y1.*2.0-x0.*x1.*z0.*z1.*2.0-y0.*y1.*z0.*z1.*2.0);

%% test
%{
N  = 1000;
p0 = 2*rand(3,N)-1;
q0 = 2*rand(3,N)-1;
p  = 2*rand(3,N)-1;
q  = 2*rand(3,N)-1;

% make approx 1% of the lines parallel
q(:,1:10:N) = p(:,1:10:N) + 5*eps(class(p0)) * rand(3,length(1:10:N));

% compare with function from matlab file exchange:
% 	http://www.mathworks.com/matlabcentral/fileexchange/...
%       29130-shortest-distance-between-two-lines-in-n-dimensions
d1 = zeros(1,N);
[d2,D2] = skewLineDist( p0, q0, p, q );
for ii = 1:N
    L1 = [ p0(:,ii), p0(:,ii) + p(:,ii) ]';
    L2 = [ q0(:,ii), q0(:,ii) + q(:,ii) ]';
    [d1(ii),~,~] = distBW2lines(L1,L2);
end
for ii = 1:10:N
    u = (p(:,ii) + q(:,ii))/2;
    v = q0(:,ii) - p0(:,ii);
    d1(ii) = norm( v - (u/norm(u)) * dot(v,u)/norm(u) );
end

tol = 100 * eps(max(max(d1(:)),max(d2(:))));
if N == 1
    disp( [d1, d2, abs(d1-d2) < tol ] );
else
    disp( all( abs(d1-d2) < tol ) );
end
%}

end

