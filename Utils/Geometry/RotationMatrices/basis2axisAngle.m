function [u,th] = basis2axisAngle( m )
%BASIS2AXISth Summary of this function goes here
%   Detailed explanation goes here

eps_rounding = 0.01; % margin to allow for rounding errors
eps_singular = 0.01; % margin to distinguish between 0 and 180 degrees
inv_sqrt_two = sqrt(0.5);
% optional check that input is pure rotation, 'isRotationMatrix' is defined at:
% http:%www.euclideanspace.com/maths/algebra/matrix/orthogonal/rotation/
if ~isRotationMatrix(m)
    error('ERROR: input matrix is not a valid rotation matrix');% for debugging
end

if ( (abs(m(1,2)-m(2,1))< eps_singular) ...
        && (abs(m(1,3)-m(3,1))< eps_singular) ...
        && (abs(m(2,3)-m(3,2))< eps_singular) )
    % singularity found
    % first check for identity matrix which must have +1 for all terms
    %  in leading diagonaland zero in other terms
    if ( (abs(m(1,2)+m(2,1)) < eps_singular) ...
            && (abs(m(1,3)+m(3,1)) < eps_singular) ...
            && (abs(m(2,3)+m(3,2)) < eps_singular) ...
            && (abs(m(1,1)+m(2,2)+m(3,3)-3) < eps_singular) )
        % this singularity is identity matrix so th = 0
        warning('WARNING: axis is arbitrary, as th of rotation is 0');
        u = [0;0;1];
        th = 0; % zero th, arbitrary axis
    else
        % otherwise this singularity is th = 180
        th = pi;
        xx = (m(1,1)+1)/2;
        yy = (m(2,2)+1)/2;
        zz = (m(3,3)+1)/2;
        xy = (m(1,2)+m(2,1))/4;
        xz = (m(1,3)+m(3,1))/4;
        yz = (m(2,3)+m(3,2))/4;
        if ((xx > yy) && (xx > zz))  % m(1,1) is the largest diagonal term
            if (xx< eps_rounding)
                x = 0;
                y = inv_sqrt_two;
                z = inv_sqrt_two;
            else
                x = sqrt(xx);
                y = xy/x;
                z = xz/x;
            end
        elseif (yy > zz)  % m(2,2) is the largest diagonal term
            if (yy< eps_rounding)
                x = inv_sqrt_two;
                y = 0;
                z = inv_sqrt_two;
            else
                y = sqrt(yy);
                x = xy/y;
                z = yz/y;
            end
        else  % m(3,3) is the largest diagonal term so base result on this
            if (zz< eps_rounding)
                x = inv_sqrt_two;
                y = inv_sqrt_two;
                z = 0;
            else
                z = sqrt(zz);
                x = xz/z;
                y = yz/z;
            end
        end
        u = [x;y;z];
    end
else
    % as we have reached here there are no singularities so we can handle normally
    s = sqrt(  (m(3,2) - m(2,3))*(m(3,2) - m(2,3)) ...
        + (m(1,3) - m(3,1))*(m(1,3) - m(3,1)) ...
        + (m(2,1) - m(1,2))*(m(2,1) - m(1,2))  ); % used to normalise
    if (abs(s) < eps_rounding)
        s=1;
    end
    % prevent divide by zero, should not happen if matrix is orthogonal and should be
    % caught by singularity test above, but I've left it in just in case
    th = acos(( m(1,1) + m(2,2) + m(3,3) - 1)/2);
    x = (m(3,2) - m(2,3))/s;
    y = (m(1,3) - m(3,1))/s;
    z = (m(2,1) - m(1,2))/s;
    u = [x;y;z];
end

[u,th] = standardAxis(u,th);

end
