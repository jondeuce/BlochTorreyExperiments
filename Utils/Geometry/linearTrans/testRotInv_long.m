if ~exist('BETA','var'); BETA=[]; end;
if ~exist('ALPHA','var'); ALPHA=[]; end;
n=10000;
% checkClouds;
% a=SCFE2_PointClouds.geo.neckCylinder{2}';
% N=null(a);
% a2=N(:,1)';
% a3=N(:,2)';
a=2*rand(1,3)-1;
a=a/norm(a);
for i=1:n
%     b=rand*a+(2*rand-1)*a2+(2*rand-1)*a3;
%     b=b/norm(b);
    b=2*rand(1,3)-1;
    b=b/norm(b);
    
    alpha=mod(atan2(b(2),b(1))-atan2(a(2),a(1)),2*pi)*180/pi;
    beta=mod(atan2(b(3),b(1))-atan2(a(3),a(1)),2*pi)*180/pi;
    
    [~,b2]=rotationInverse(a,alpha,beta);
    
	alpha2=mod(atan2(b2(2),b2(1))-atan2(a(2),a(1)),2*pi)*180/pi;
    beta2=mod(atan2(b2(3),b2(1))-atan2(a(3),a(1)),2*pi)*180/pi;
    
    if abs(alpha-alpha2)>1e-9 || abs(beta-beta2)>1e-9
        disp(['Alpha: ' num2str(alpha,8)]);
        disp(['Beta: ' num2str(beta,8)]);
        ALPHA=[ALPHA;alpha];
        BETA=[BETA;beta];
        
        [~,b2]=rotationInverse(a,alpha,beta,true);
    end
    
    if mod(i,100)==0
        disp(['i: ' num2str(i) ' ' 'beta: ' mat2str([max(BETA(BETA<180)) min(BETA(BETA>180))-360],4)])
    end
end