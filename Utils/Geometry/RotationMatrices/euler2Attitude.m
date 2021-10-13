function H_I_to_B = euler2Attitude( psi, th, phi )
%EULER2ATTITUDE Attitude matrix based on definitions at
%   https://www.princeton.edu/~stengel/MAE331Lecture9.pdf

H_I_to_1 = ...
    [   cos(psi)    sin(psi)    0
       -sin(psi)    cos(psi)    0
        0           0           1       ];
    
H_1_to_2 = ...
    [   cos(th)     0          -sin(th)
        0           1           0
        sin(th)     0           cos(th) ];
    
H_2_to_B = ...
    [   1           0           0
        0           cos(phi)    sin(phi)
        0          -sin(phi)    cos(phi) ];
    
H_I_to_B = H_2_to_B * H_1_to_2 * H_I_to_1;
    
% H_I_to_B = ...
%     [                              cos(psi)*cos(th),                              cos(th)*sin(psi),         -sin(th)
%       cos(psi)*sin(phi)*sin(th) - cos(phi)*sin(psi), cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(th), cos(th)*sin(phi)
%       sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(th), cos(phi)*sin(psi)*sin(th) - cos(psi)*sin(phi), cos(phi)*cos(th) ];
 

end

