function decay_curve = EPGdecaycurve(ETL,flip_angle,TE,T2,T1,refcon)
% Computes the normalized echo decay curve for a MR spin echo sequence with the given parameters.
%
% ETL: Echo train length (number of echos)
% flip_angle: Angle of refocusing pulses (degrees)
% TE: Interecho time (seconds)
% T2: Transverse relaxation time (seconds)
% T1: Longitudinal relaxation time (seconds)
% refcon: Value of Refocusing Pulse Control Angle

% Initialize magnetization phase state vector (MPSV) and set all
% magnetization in the F1 state.
M=zeros(3*ETL,1);
M(1,1)=exp(-(TE/2)/T2)*sin(flip_angle*(pi/180)/2);
% Compute relaxation matrix
T_r=relaxmat(ETL,TE,T2,T1);
% Initialize vector to track echo amplitude
echo_amp=zeros(1,ETL);
% Compute flip matrix
[T_1,T_p]=flipmat(flip_angle*(pi/180),ETL,refcon);
% Apply first refocusing pulse and get first echo amplitude
M(1:3)=T_1*M(1:3);
echo_amp(1,1)=abs(M(2,1))*exp(-(TE/2)/T2);
% Apply relaxation matrix
M=T_r*M;
% Perform flip-relax sequence ETL-1 times
for x=2:ETL
    % Perform the flip
    M=T_p*M;
    % Record the magnitude of the population of F1* as the echo amplitude
    % and allow for relaxation
    echo_amp(1,x)=abs(M(2,1))*exp(-(TE/2)/T2);
    % Allow time evolution of magnetization between pulses
    M=T_r*M;
end
decay_curve=echo_amp;
end
%==========================================================================
%==========================================================================
function [T_1,T_p] = flipmat(alpha,num_pulses,refcon)
% Computes the transition matrix that describes the effect of the refocusing
% pulse on the magnetization phase state vector.

% Compute the flip matrix as given by Hennig (1988), but corrected by Jones
% (1997)
T_1=[cos(alpha/2)^2,sin(alpha/2)^2,-1i*sin(alpha);...
    sin(alpha/2)^2,cos(alpha/2)^2,1i*sin(alpha);...
    -0.5i*sin(alpha),0.5i*sin(alpha),cos(alpha)];
alpha2=alpha*refcon/180;
T_2=[cos(alpha2/2)^2,sin(alpha2/2)^2,-1i*sin(alpha2);...
    sin(alpha2/2)^2,cos(alpha2/2)^2,1i*sin(alpha2);...
    -0.5i*sin(alpha2),0.5i*sin(alpha2),cos(alpha2)];
% Create a block matrix with T_1 on the diagonal and zeros elsewhere
% T_p=spalloc(3*num_pulses,3*num_pulses,9*num_pulses);
% for x=1:num_pulses
%     T_p(3*x-2:3*x,3*x-2:3*x)=T_2;
% end
x=kron(ones(3*num_pulses,1),[1,2,3]')+kron((0:3:3*num_pulses-1)',ones(9,1));
y=kron((1:3*num_pulses)',ones(3,1));
v=kron(ones(num_pulses,1),T_2(:));

T_p=sparse(x,y,v,3*num_pulses,3*num_pulses);

end
%==========================================================================
%==========================================================================
function T_r = relaxmat(num_states,te,t2,t1)
% Computes the relaxation matrix that describes the time evolution of the
% magnetization phase state vector after each refocusing pulse.

% Create a matrix description of the time evolution as described by
% Hennig (1988)

%T_r=zeros(3*num_states,3*num_states);

E2=exp(-te/t2);
E1=exp(-te/t1);
x=zeros(3*num_states-1,1);
y=zeros(3*num_states-1,1);
v=zeros(3*num_states-1,1);

% F1* --> F1
%T_r(1,2)=exp(-te/t2);
x(1)=1;
y(1)=2;
v(1)=E2;
% F(n)* --> F(n-1)*
%for x=1:num_states-1
%    T_r(3*x-1,3*x+2)=exp(-te/t2);
%end
x(2:num_states)=2:3:3*num_states-4;
y(2:num_states)=5:3:3*num_states-1;
v(2:num_states)=E2*ones(num_states-1,1);
% F(n) --> F(n+1)
%for x=1:num_states-1
%    T_r(3*x+1,3*x-2)=exp(-te/t2);
%end
x(num_states+1:2*num_states-1)=4:3:3*num_states-2;
y(num_states+1:2*num_states-1)=1:3:3*num_states-5;
v(num_states+1:2*num_states-1)=E2*ones(num_states-1,1);
% Z(n) --> Z(n)
%for x=1:num_states
%    T_r(3*x,3*x)=exp(-te/t1);
%end
x(2*num_states:3*num_states-1)=3:3:3*num_states;
y(2*num_states:3*num_states-1)=3:3:3*num_states;
v(2*num_states:3*num_states-1)=E1*ones(num_states,1);

T_r=sparse(x,y,v,3*num_states,3*num_states);
end%==========================================================================