function [u,th] = standardAxis( u, th, tol )
%STANDARD_AXIS ensures that the rotation axis u and angle th are in
%standard form, i.e. u is chosen to be in the upper half plane.

if nargin < 3
    tol = 0.001;
end

u = u(:)/norm(u);
if abs(u(3)) > tol
    if u(3) < 0
        u = -u;
        th = -th;
    end
else
    if abs(u(1)) > tol
        if u(1) < 0
            u = -u;
            th = -th;
        end
    else
        if u(2) < 0
            u = -u;
            th = -th;
        end
    end
end

end

