function [ w ] = testomega(x,y)
%OMEGA Summary of this function goes here
%   Detailed explanation goes here

B0    = -3.0;         % External magnetic field (z-direction) [T]
gamma = 2.67515255e8; % Gyromagnetic ratio [rad/s/T]
w0    = gamma * B0;   % Resonance frequency [rad/s]
th    = pi/2;         % Main magnetic field angle w.r.t B0 [rad]
c2    = cos(th)^2;
s2    = sin(th)^2;
ChiI  = -60e-9; % Isotropic susceptibility of myelin [ppb] (check how to get it) (Xu et al. 2017)
ChiA  = -120e-9; % Anisotropic Susceptibility of myelin [ppb] (Xu et al. 2017)
E     =  10e-9; % Exchange component to resonance freqeuency [ppb] (Wharton and Bowtell 2012)

% % Testing
% ChiI  = 0;
% ChiA  = 1;
% E     = 0;
% w0    = 1;

g  = 0.8;
ro = 0.5;  ro2 = ro^2;
ri = g*ro; ri2 = ri^2;

r2 = x.^2 + y.^2;
r = sqrt(r2);
t = atan2(y,x);
w = zeros(size(x));

b = (r < ri);
w(b) = w0 * ChiA * 3*s2/4 * log(ro/ri);

b = (ri <= r & r <= ro);
w(b) = ...
    w0 * ChiI * (1/2) * (c2 - 1/3 - s2 * cos(2*t(b)) .* (ri2./r2(b))) + ...
    w0 * E + ...
    w0 * ChiA * (s2 * (-5/12 - cos(2*t(b))/8 .* (1+ri2./r2(b)) + (3/4) * log(ro./r(b))) - c2/6);

b = (ro < r);
w(b) = ...
    w0 * ChiI * (s2/2) * cos(2*t(b)) .* (ro2 - ri2) ./ r2(b) + ...
    w0 * ChiA * (s2/8) * cos(2*t(b)) .* (ro2 - ri2) ./ r2(b);

end

