%% Initialize
Gsize = [512,512,512];
% Gsize = [256,256,256];
% Gsize = [16,16,16];
x = randnc(Gsize);
Diag = randnc(Gsize);
Darray = randn(Gsize);
Dcoeff = abs(randn(1));

A = BlochTorreyOp(Diag,Darray,Gsize,Gsize,true); %anisotropic
B = BlochTorreyOp(Diag,Dcoeff,Gsize,Gsize,true); %isotropic

%% Time
N = 3;
ts = zeros(N,1);
for ii = 1:N
    ts(ii) = timeit(@()A*x);
    display_toc_time(ts(ii), sprintf('Anisotropic: %d/%d', ii, N));
end
disp(min(ts))

%% Time
N = 3;
ts = zeros(N,1);
for ii = 1:N
    ts(ii) = timeit(@()B*x);
    display_toc_time(ts(ii), sprintf('Isotropic: %d/%d', ii, N));
end
disp(min(ts))
