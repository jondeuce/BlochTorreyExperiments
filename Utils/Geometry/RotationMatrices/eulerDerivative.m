function J = eulerDerivative( th, phi, w, flag )
%EULERDERIVATIVE Summary of this function goes here
%   Detailed explanation goes here

% L_B_to_I = ...
%     [   1   sin(phi)*tan(th)    cos(phi)*tan(th)
%         0   cos(phi)           -sin(phi)
%         0   sin(phi)/cos(th)    cos(phi)/cos(th)    ];

cp = cos(phi);
sp = sin(phi);
ct = cos(th);
st = sin(th);

L_I_to_B = ...
    [   1   0          -st
        0   cp          sp*ct
        0  -sp          cp*ct    ];

L_B_to_I = ...
    [   1   sp*st/ct    cp*st/ct
        0   cp         -sp
        0   sp/ct       cp/ct    ];

if nargin < 4
    J = L_B_to_I * w;
    % psi_dot = J(3);
    % th_dot  = J(2);
    % phi_dot = J(1);
else
    if flag
        J = L_I_to_B * w;
    else
        J = L_B_to_I * w;
    end
end

end

