function [omega,omega_filtered] = test_orientation( test )
%TEST_FILTER Summary of this function goes here
%   Detailed explanation goes here

close all;

if nargin < 1; test = 'basis'; end

%% read data
fileID = fopen('gyro_accel_data.txt','r');
% fileID = fopen('gyro_data_motors_on.txt','r');

A = fscanf(fileID,'%f\t%f\t%f\n');
A = reshape(A,[3,length(A)/3])';
% A = fscanf(fileID,'%f\t%f\t%f\t%f\t%f\t%f\n');
% A = reshape(A,[6,length(A)/6])';

start = 146;
stop  = 1951;
A = A(start:stop,:);

omega = A(:,1:3);
% acc = A(:,4:6);

fclose(fileID);

%% execute a test
switch upper(test)
    case upper('basis') % basis test
        [omega,omega_filtered] = basis_test(omega);
    otherwise
        [omega,omega_filtered] = basis_test(omega);
end

end

function [omega,omega_filtered] = basis_test(omega)

%% get time data
Fs = 125; %sampling freq [Hz]
Ts = 1/Fs; %sampling time [s]
T0 = 30; %memory of iir averaging [s]
% L = 2^floor(log2(size(omega,1)));
% omega = omega(1:L,:);
L = size(omega,1);
t = (0:L-1)'*Ts;

plot_time = true;
% plot_time = false;
% plot_freq = true;
plot_freq = false;

%% add discontinuity
i1 = floor( 0.4 * length(omega) );
i2 = floor( 0.6 * length(omega) );

rect_bump = zeros(size(omega));
rect_bump(i1:i2,:) = 1;

i3 = round((i1+i2)/2);
ramp_bump = zeros(size(omega));
ramp_bump(i1:i3,:)   = repmat(linspace(0,1,i3-i1+1)',[1,size(omega,2)]);
ramp_bump(i3+1:i2,:) = repmat(linspace(1,0,i2-i3)',[1,size(omega,2)]);

% omega = omega + 5*rect_bump;
% omega = omega + 5*ramp_bump;
% omega = omega + 0.5*rect_bump;
% omega = omega + 0.5*ramp_bump;

%% filter signals
omega_filtered = zeros(size(omega));

% initial data
mu = mean(omega(1:floor(0.2*size(omega,1)),:));

for i = 1:3    
    %% get filtered data
    % type = 'bessel_order_5_a1_008_a2_2'; param = [];
    % type = 'bessel_order_5_a1_016_a2_4'; param = [];
    % type = 'bessel_order_2_a1_0008_a2_12'; param = [];
    % type = 'bessel_order_5_a2_4'; param = [];
    % type = 'bessel_order_5_a2_04'; param = [];
    % type = 'bessel_order_5_a2_024'; param = [];
    % type = 'butter_order_5_a2_004'; param = [];
    % type = 'integrator'; param = [];
    % type = 'median'; param = 7;
    % type = 'average'; param = 10;
    % type = 'identity'; param = [];
%     [filt1,v1] = get_filter('median',7,mu(i)); % median starts at average value
    [filt1,v1] = get_filter('iir_average',Ts/0.1,mu(i));
    % [filt1,v1] = get_filter('identity');
    % [filt2,v2] = get_filter('iir_dc_level',Ts/T0,inf,mu(i)); % dc_leveler starts at average value
    [filt2,v2] = get_filter('bessel_order_5_a2_04');
%     [filt2,v2] = get_filter('identity');
    % [filt3,v3] = get_filter('bessel_order_5_a2_04');
    [filt3,v3] = get_filter('iir_dc_level',Ts/T0,0.5,mu(i)); % dc_leveler starts at average value
    % [filt3,v3] = get_filter('identity');
    % [filt3,v3] = get_filter('linear',7);
    % [filt3,v3] = get_filter('compound',5);
%     [filt3,v3] = get_filter('identity');    
%     [filt3,v3] = get_filter('median',7,mu(i)); % median starts at average value
    
    % filter signals
    omega_filtered(:,i) = filter_signal(omega(:,i),filt1,v1);
    omega_filtered(:,i) = filter_signal(omega_filtered(:,i),filt2,v2);
    omega_filtered(:,i) = filter_signal(omega_filtered(:,i),filt3,v3);
end

%% plot results
labels = {'omega_x','omega_y','omega_z'};
for i = 1:3
    if plot_time
        plot_before_after(t*1000, omega(:,i), omega_filtered(:,i), ...
            [labels{i} ' before'], [labels{i} ' after'], ...
            'time (ms)', [labels{i} ' (rad/s)'], [labels{i} ' vs. time']);
    end
    if plot_freq
        [Y1,X] = get_fft(omega(:,i),Fs,L);
        [Y2,~] = get_fft(omega_filtered(:,i),Fs,L);
        Y1(1) = 0; Y2(1) = 0; % zero out dc modes
        plot_before_after(X, Y1, Y2, ...
            [labels{i} ' before'], [labels{i} ' after'], ...
            'freq (Hz)', [labels{i} ' (rad/s)'], [labels{i} ' vs. freq']);
    end
end

end

function Y = filter_signal(y,filt,v)

yy = 0;
Y  = zeros(size(y));
for i = 1:length(y)
%     if y(i) > 2
%         ;
%     end
    [yy,v] = filt(y(i),v,yy);
    Y(i) = yy;
end

end

function [Y,X] = get_fft(y,Fs,L)

y = fft(y,L);
A = abs(y/L);
Y = A(1:L/2+1);

Y(2:end-1) = 2*Y(2:end-1);
X = Fs*(0:(L/2))/L;

end

function h = plot_before_after(x,y1,y2,leg1,leg2,xtitle,ytitle,plotlabel)

figure, hold on, grid on
hh = plot(x,[y1,y2],'linewidth',3);
legend(hh,leg1,leg2);

xlabel(xtitle,'fontsize',12,'fontweight','bold');
ylabel(ytitle,'fontsize',12,'fontweight','bold');
title(plotlabel,'fontsize',14,'fontweight','bold');

if nargout > 0; h = hh; end

end

function h = plot_fft(y,Fs,L)
[Y,X] = get_fft(y,Fs,L);
h = plot(X,Y);
end

function h = plot_semilog_fft(y,Fs,L)
[Y,X] = get_fft(y,Fs,L);
h = plot(log10(X),Y);
end