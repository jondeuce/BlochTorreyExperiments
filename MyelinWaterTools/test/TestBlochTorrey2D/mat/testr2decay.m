function [ R2 ] = testr2decay(x,y)
%OMEGA Summary of this function goes here
%   Detailed explanation goes here

R2_sp = 1/15e-3; % Relaxation rate of small pool [s^-1] (Myelin) (Xu et al. 2017) (15e-3s)
R2_lp = 1/63e-3; % Relaxation rate of large pool [s^-1] (Intra/Extra-cellular)

g  = 0.8;
ro = 0.5;  ro2 = ro^2;
ri = g*ro; ri2 = ri^2;

r2 = x.^2 + y.^2;
R2 = zeros(size(x));

b = (ri2 <= r2 & r2 <= ro2);
R2( b) = R2_sp;
R2(~b) = R2_lp;

end

