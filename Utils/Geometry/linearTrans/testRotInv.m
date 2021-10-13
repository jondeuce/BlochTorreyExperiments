%% look for optimal number of divisions during search


%% check if rotationInverse works
% checkClouds;
% a=SCFE2_PointClouds.geo.neckCylinder{2}';
% N=null(a);
% a2=N(:,1)';
% a3=N(:,2)';
% 
% b=rand*a+(2*rand-1)*a2+(2*rand-1)*a3;
% b=b/norm(b);
% 
% alpha=mod(atan2(a(2),a(1))-atan2(b(2),b(1)),2*pi)*180/pi;
% beta=mod(atan2(a(3),a(1))-atan2(b(3),b(1)),2*pi)*180/pi;
% 
% [~,b2,diff]=rotationInverse(a,alpha,beta,true);