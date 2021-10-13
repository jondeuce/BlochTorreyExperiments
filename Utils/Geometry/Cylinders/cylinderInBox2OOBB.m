function [ box ] = cylinderInBox2OOBB( p, vz, r, B, B0 )
%CYLINDERINBOX2OOBB Gets OOBB for cylinder (p,vz,r) contained in the box
% with dimensions B and center B0. Output has size [3 x 8 x NumCylinders].
% 
%                  8 __________ 7
%                   /|        /| 
%                  / |       / | 
%               5 /__|______/ 6| 
%                 |  |  /\  |  | 
%                 |  | /vz\ |  | 
%                 |  |  ||  |  | 
%                 |  |  ||  |  | 
%                 |  |  p.  |  | 
%                 |4 |______|__|3
%                 | /       | /
%                 |/________|/ 
%               1             2

[tmin, tmax, ~, ~, p, Pmin, Pmax]	=	rayBoxIntersection( p, vz, B, B0 );

D           =   2 * r; % cyl. diameter
L           =   tmax - tmin; % cyl. length from center to center
Lhalf       =   L/2;

Nbot        =   zeros(3,numel(r)); % normal vector at bottom cyl intersection
Nbot(Pmin)  =   1;
Ntop        =   zeros(3,numel(r)); % normal vector at top cyl intersection
Ntop(Pmax)  =	1;

vz          =   unit(vz,1);
[th1,th2]	=   deal( minAngle3D(vz,Nbot), minAngle3D(vz,Ntop) );

if th1 > th2
    [vx,vy]	=	get_nullvecs(vz,Nbot);
else
    [vx,vy]	=	get_nullvecs(vz,Ntop);
end

% add length contributions from glancing incidence at top and bottom ends
theta       =   max(th1,th2);
Lhalf       =   Lhalf + D .* tan(theta); 

Lvz         =   bsxfun( @times, Lhalf, vz );
box         =   bsxfun( @plus,	...
    reshape( p,	3, 1, [] ),     ...
    reshape( [	bsxfun(@times,r,-vx-vy) - Lvz;      ...
                bsxfun(@times,r, vx-vy) - Lvz;      ...
                bsxfun(@times,r, vx+vy) - Lvz;      ...
                bsxfun(@times,r,-vx+vy) - Lvz;      ...
                bsxfun(@times,r,-vx-vy) + Lvz;      ...
                bsxfun(@times,r, vx-vy) + Lvz;      ...
                bsxfun(@times,r, vx+vy) + Lvz;      ...
                bsxfun(@times,r,-vx+vy) + Lvz	],  ...
                3, 8, [] )      ...
        );

end

function [vx,vy] = get_nullvecs(vz,N)

vx	=   cross(N,vz,1);
if norm(vx) > 100 * eps(class(vx))
    vx	=	unit(vx,1); % vx = N x vz
    vy	=	unit(cross(vz,vx,1),1);  % vy = (vz x N) x vz
else
    [vx,vy,~]	=   nullVectors3D(vz);
end

end