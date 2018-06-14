%% Initialize
% Gsize = [256,256,256];
Gsize = [512,512,512];
x = randnc(Gsize);
D = randn(Gsize);
Diag = randnc(Gsize);

A = BlochTorreyOp(Diag,D,size(D),size(D),true);
disp(A);

%% Time
N = 3;
ts = zeros(N,1);
for ii = 1:N
    ts(ii) = timeit(@()A*x);
    display_toc_time(ts(ii), sprintf('%d/%d', ii, N));
end
disp(min(ts))
