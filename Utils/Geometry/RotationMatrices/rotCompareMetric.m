function m = rotCompareMetric
%ROTCOMPAREMETRIC Summary of this function goes here
%   Detailed explanation goes here

n = 1000;
theta = pi*rand(n,1);
metric = zeros(n,1);
for i = 1:n
    [R1,~] = qr(rand(3)-0.5);
    u = rand(3,1);
    u = u/norm(u);
    R2 = axisAngle2Rot(u,theta(i));
    metric(i) = comp(R1,R2*R1);
end
        
% Plot error metric vs. theta
%   -function is exactly given by sqrt(1-cos(theta/2)^2) ~ theta/2 for
%   small theta
close all
figure, hold on, grid on
plot(theta,metric,'o');
m = max(metric);

end

function m = comp(R1,R2)

d = dot(R1(:,1),R2(:,1)) + dot(R1(:,2),R2(:,2)) + dot(R1(:,3),R2(:,3));
m = sqrt(3)/2*sqrt(1-d/3);

end
