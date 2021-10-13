function [R] = randRotMat(N)
%RANDROTMAT Returns a random rotation matrix of dimension N (default 3).

if nargin < 1 || isempty(N)
    N = 3;
end

if N == 3
    % convert random quaternion to rotation matrix
    q = randn(4,1);
    q = q/norm(q); % * sign(q(1)); % don't need to fix sign
    
    ww = q(1) * q(1);
    xx = q(2) * q(2);
    yy = q(3) * q(3);
    zz = q(4) * q(4);
    xy = q(2) * q(3);
    zw = q(1) * q(4);
    xz = q(2) * q(4);
    yw = q(3) * q(1);
    yz = q(3) * q(4);
    xw = q(1) * q(2);

    % initialize rotation part
    R = [ ww + xx - yy - zz, 2 * (xy - zw),     2 * (xz + yw)
          2 * (xy + zw),     ww - xx + yy - zz, 2 * (yz - xw)
          2 * (xz - yw),     2 * (yz + xw),     ww - xx - yy + zz ];
else
    R = orth(randn(N,N));
    
    if abs( det(R) - 1 ) >=  1000 * eps(max(abs(R(:))))
        R(:,end) = -R(:,end);
    end
end

end

%% Old direct method - not worth
% function [varargout] = randRotMat(n,PsiMax,dim)
% %RANDROTMAT returns a set of n random 2x2 or 3x3 rotation matrices
% 
% if nargin < 1 || isempty(n)
%     n	=	1;
% end
% 
% if nargin < 3 || isempty(dim)
%     dim	=	3;
% end
% 
% if nargin < 2 || isempty(PsiMax)
%     if dim == 2,	PsiMax	=   2*pi;
%     else            PsiMax	=   pi;
%     end
% end
% 
% switch dim
%     case 2
%         [varargout{1:2}]	=   randRotMat2D(n,PsiMax);
%     otherwise
%         [varargout{1:3}]	=	randRotMat3D(n,PsiMax);
% end
% 
% end
% 
% function [Rot,psi] = randRotMat2D(n,PsiMax)
% 
% Rot     =	zeros(2,2,n);
% psi     =   PsiMax * rand(1,n);
% for ii = 1:n
%     [s,c]       =   deal( sin(psi(ii)), cos(psi(ii)) );
%     Rot(:,:,ii)	=   [	c, -s
%                         s,	c	];
% end
% 
% end
% 
% function [Rot,u,psi] = randRotMat3D(n,PsiMax)
% 
% psi_pdf	=   @(x) 1/pi*( x - sin(x) );
% Pmax	=   psi_pdf(PsiMax);
% 
% % th      =	2 * pi * rand(1,n);
% % phi     =	acos( 2 * rand(1,n) - 1 );
% % u       =	[ cos(th).*sin(phi); sin(th).*sin(phi); cos(phi) ];
% u       =   2 * rand(3,n) - 1;
% u       =   bsxfun( @rdivide, u, sqrt( sum( u.^2, 1 ) ) );
% 
% Rot     =	zeros(3,3,n);
% psi     =   zeros(1,n);
% for ii = 1:n
%     p           =   Pmax * rand;
%     psi(ii)     =   fzero( @(x) p - psi_pdf(x), [0,PsiMax] );
%     Rot(:,:,ii)	=   axisAngle2Rot( u(:,ii), psi(ii) );
% end
% 
% end

%% This method does not produce the right amount of near-identity matrices
% % cdf of angle for unif rand rotation matrix for 0<=th<=pi is
% % (1/pi)*(psi-sin(psi)), and the pdf is (1/pi)*(1-cos(psi))
% p = rand;
% psi = sqrt(p*pi^2);
% if psi > 1e-3
%     psi = psi - ((psi-sin(psi))/pi-p)/((1-cos(psi))/pi);
%     psi = psi - ((psi-sin(psi))/pi-p)/((1-cos(psi))/pi);
%     psi = psi - ((psi-sin(psi))/pi-p)/((1-cos(psi))/pi);
% end
% u = 2*rand(3,1)-1;
% u = u / norm(u);
% Rot = axisAngle2Rot(u,psi);

%% test script
%{
figure, hold on, grid on
for i=1:1000
    R = randRotMat;
    plot3(R(:,1)',R(:,2)',R(:,3)','.');
end
%}