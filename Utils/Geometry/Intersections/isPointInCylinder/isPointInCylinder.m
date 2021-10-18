function [ b ] = isPointInCylinder( Points, p, vz, r, vx, vy, isUnit )
%ISPOINTINCYLINDER Returns boolean list which is true if the corresponding
% point in Points is in any of the cylinders defined by p, v, and r, and
% false otherwise
%
% INPUT ARGUMENTS
%   Points:	[3xN]	Array of points to check
%   p:      [3xP]	Point on the cylinder
%   vz:     [3xP]	Direction of cylinder axis
%   r:      [3xP]	Radius of cylinder
%   vx:     [3xP]   (optional) First right-handed basis vector to vz
%   vy:     [3xP]   (optional) Second right-handed basis vector to vz
%   isUnit: [T/F]   (optional) True indicates that the inputs vx, vy, vz
%                   are already all unit vectors and need not be normalized
% 
% NOTE:     If provided, each V = [vx(i,:), vy(i,:), vz(i,:)] should form a
%           proper orthogonal transformation, i.e. det(V) = 1 and V*V' = I
%               -e.g. if vz is [3x1] and vxy = null(vz'), then one of
%               [vxy,vz] or [fliplr(vxy),vz] will satisfy this condition
%
% OUTPUT ARGUMENTS
%   b:      [1xN]	Boolean array; true if point is in any cylinder,
%                   otherwise false

%% Parse inputs
useProjectMethod	=   false;
if nargin < 6 || ( isempty(vx) || isempty(vy) )
    useProjectMethod	=   true;
end

if nargin < 7 || isempty(isUnit)
    isUnit	=   false;
end

if ~isUnit
    if ~useProjectMethod
        vx	=   unit(vx,1);
        vy	=   unit(vy,1);
    end
    vz	=   unit(vz,1);
end

%% Initialize
N	=   size(Points,2);
P	=   size(p,2);

if useProjectMethod
%% Use projection method
    
    b	=   true(1,N);
    for ii = 1:P
        % Get rejection (shortest vectors from line L(t)=p+v*t to Points)
        %Rej	=   Points(:,b);
        %Rej	=	Rej - projPointLine( Rej, vz(:,ii), p(:,ii), 1 );
        Rej	=	Points;
        Rej	=   Rej - projPointLine( Points, vz(:,ii), p(:,ii), 1 );
        
        % If rejection is shorter than cylinder radius, point is in cylinder
        %b(b)	=	b(b) & ( sum( Rej.^2, 1 ) > r(ii)^2 );
        b       =	b & ( sum( Rej.^2, 1 ) > r(ii)^2 );
        if ~any(b)
            break
        end
    end
    b	=   ~b;
    
else
%% Use transformation method
    
    % work in column basis for speed
    [p,vx,vy,vz]	=   deal( p', vx', vy', vz' );
    Pb	=	Points';
    b	=	true(N,1);
    
    for ii = 1:P
        % Get rejection (shortest vectors from line L(t)=p+vz*t to Points)
        V_T	=   [vx(ii,:); vy(ii,:); vz(ii,:)]';
        Pb	=	bsxfun( @minus, Pb, p(ii,:) );
        Pb	=   Pb * V_T;
        
        % Transform cylinder axes/points as well
        p	=   bsxfun( @minus, p, p(ii,:) );
        [p,vx,vy,vz]	=   deal( p*V_T, vx*V_T, vy*V_T, vz*V_T );
        
        % If rejection is shorter than cylinder radius, point is in cylinder
        b0	=   ( Pb(:,1).^2 + Pb(:,2).^2 > r(ii)^2 );
        b	=	b & b0;
        if ~any(b)
            break
        end
    end
    b	=   ~b';
    
end


end

