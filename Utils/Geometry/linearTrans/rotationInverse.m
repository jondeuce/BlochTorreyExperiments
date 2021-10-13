function [R_inv, v] = rotationInverse( v0, alpha, beta, istest )
%ROTATIONINVERSE returns the matrix R_inv such that v=v0*R_inv (v0 a row
%vector, R_inv a 3 x 3 rotation matrix) produces v such that:
%   -the projections of v0 and v onto the xy-plane form the angle 'alpha'
%   when measured counterclockwise from the positive x-axis
%   -the projections of v0 and v onto the xz-plane form the angle 'beta'
%   when measured counterclockwise from the positive x-axis
% 
% Input:
%   -three dimensional vector 'v0'
%   -angle 'alpha', alpha (degrees) (will be mapped to [0,360) )
%   -angle 'beta', beta (degrees) (will be mapped to [0,360)
% 
% NOTE: alpha, beta should be in degrees

% For reference: definition of signed axial (alpha) and coronal (beta)
% angles (degrees) for 3D vectors v0, v1
% a=@(v1,v0) (atan2(v1(2),v1(1))-atan2(v0(2),v0(1)))*(180/pi);
% c=@(v1,v0) (atan2(v1(3),v1(1))-atan2(v0(3),v0(1)))*(180/pi);

if nargin==3; istest=false; end;

alpha=mod(alpha,360)*(pi/180);
beta=mod(beta,360)*(pi/180);

v0=(v0(:))';
Rz=[cos(alpha) sin(alpha) 0; -sin(alpha) cos(alpha) 0; 0 0 1];

% rotate v1 about z-axis by alpha; alpha angle set
v1=v0*Rz;

% calculate the vector u normal to the plane spanned by the unit vector
% k = [0 0 1] and the projection of v1 onto the xy-plane
%   -rotation about u will not change the alpha angle
k=[0 0 1];
v1k=v1; v1k(3)=0;
u=cross(k,v1k);
u=u/norm(u);

% get rotation matrix about u-axis (note by definition of u above u(3)==0)
% as a function of the angle of rotation t
% http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
Ru=@(t) cos(t)*eye(3) - sin(t)*[0,-u(3),u(2);u(3),0,-u(1);-u(2),u(1),0] + (1-cos(t))*(u')*u;

% project v0 onto xz plane and convert into representative 2D vector in
% polar coordinates
V=norm(v0([1 3]));
r0_j=[ mod(atan2(v0(3)/V,v0(1)/V),2*pi); 1 ];

% solve for theta to set beta angle
EPS=0.01*pi/180;
if (abs(beta)<EPS || abs(2*pi-beta)<EPS) && ~istest
    R_inv=Rz;
    if nargout>1; v=v1; end;
    return;
end

% get bounds t0 s.t. t0(1) < t_zero < t0(2)
num_div=4;
t0=get_t0(0,2*pi*(num_div-1)/num_div,r0_j,v1,Ru,beta,num_div);

if t0(1)<t0(2)
    [T,fval,exitflag]=fzero(@(t) fun(r0_j,v1,Ru(t),beta),[t0(2) t0(1)+2*pi]);
    theta=mod(T,2*pi);
else
    [theta,fval,exitflag]=fzero(@(t) fun(r0_j,v1,Ru(t),beta),[t0(1) t0(2)]);
end

if istest; testplot(r0_j,v1,Ru,beta,t0,theta,fval,exitflag); end;

% Return inverse matrix R_inv, and vector v s.t. v = R_inv * v0, with v and
% v0 having alpha and beta angles as input
R_inv=Rz*Ru(theta);
if nargout>1; v=v0*R_inv; end;

end

function diff = fun(r0_j,v1,Ru,beta)

v1=v1*Ru;
r1_j=[ mod(atan2(v1(3),v1(1)),2*pi); hypot(v1(1),v1(3)) ];
r1_j(1)=mod(r1_j(1)-r0_j(1),2*pi);

diff=r1_j(1)-beta;

end

function t0=get_t0(t_start,t_end,r0_j,v1,Ru,beta,num_div)

if num_div<=3
    error('Error: require at least 4 divisions to detect sign change');
end

if t_start<t_end
    T=linspace(t_start,t_end,num_div);
else
    T=mod(linspace(t_start,t_end+2*pi,num_div),2*pi);
end
f=zeros(length(T),1);
count=1;
for t=T
    f(count)=fun(r0_j,v1,Ru(t),beta);
    if count==4;
        df=diff(f(1:4));
        bf=(sign(df)==1);
        if sum(bf)>=2; sgn=1;
        else sgn=-1;
        end
        ind=find(sign(df)~=sgn,1);
        if ~isempty(ind); break; end;
    elseif count>4
        if sign(f(count)-f(count-1))~=sgn
            ind=count-1;
            break;
        end
    end
    count=count+1;
end

if count==num_div+1
    if sign(f(1)-f(end))~=sgn; bound1=length(T); bound2=1; end;
else
    bound1=ind; bound2=ind+1;
end

if sign(f(bound1))~=sign(f(bound2))
    t0=T([bound1 bound2]);
    return;
else
    t0=get_t0(T(bound1),T(bound2),r0_j,v1,Ru,beta,num_div);
end

end

function testplot(r0_j,v1,Ru,beta,t0,theta,fval,exitflag)

fig=9;
g=get(0,'children');
if ismember(fig,g); clf(fig,'reset'); end;
figure(fig)
hold on

lw=2;
[x1,y1]=fplot(@(t) fun(r0_j,v1,Ru(t*(pi/180)),beta),[0 360]);
plot(x1,y1,'b--','LineWidth',lw);

[x0,y0]=fplot(@(t) 0,[0 360]);
plot(x0,y0,'r','LineWidth',lw);

line(repmat(mean(t0)*180/pi,[1 2]),ylim,'Color','r','LineWidth',lw);
line([theta theta]*180/pi,ylim,'Color','r','LineWidth',lw);

hold off;
grid on;
axis([-10 370 min(y1)-1 max(y1)+1]);

disp(['Theta: ' num2str(theta*180/pi,4)])
disp(['Discontinuity: ' num2str(mean(t0)*180/pi,4)])
disp(['Difference: ' num2str(abs(theta-mean(t0))*180/pi,4)])
disp(['Func val: ' num2str(fval)])
disp(['Exit Flag: ' num2str(exitflag)])
disp(['Max - Min: ' num2str(max(y1)-min(y1),4)]);

drawnow;
pause(5);

end