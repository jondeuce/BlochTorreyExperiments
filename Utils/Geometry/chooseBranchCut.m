function theta = chooseBranchCut( theta, theta0, units )
%CHOOSEBRANCHCUT Choose a branch cut for a set of angles.
%   THETA = CHOOSEBRANCHCUT(THETA,THETA0,UNITS) Translates the angles THETA
%   to the the branch cut at THETA0 inclusive, such that all angles now lie
%   in [THETA0,THETA0+2pi) (radians) or [THETA0,THETA0+360) (degrees).
% 
%   Inputs:
%       -THETA: input angles
%       -THETA0: start of branch cut
%       -UNITS ('radians'): can be 'radians' or 'degrees'
%   Outputs:
%       -THETA: output angles at the new branch cut

if nargin < 3 % skip the switch statement for speed
    theta = mod(theta - theta0, 2*pi) + theta0;
    return
end

switch upper(units)
    case 'DEGREES'
        theta = mod(theta - theta0, 360) + theta0;
    otherwise
        theta = mod(theta - theta0, 2*pi) + theta0;
end

end

