%%%%
%%%% lsqnonneg_reg
%%%%

n = 100;
C = exp(sin(reshape((1:n^2)', n, n)));
y = sin((1:n)');
d = C*y;
Chi2Factor = 1.02;
x = lsqnonneg_reg(C, d, Chi2Factor);

timeit(@() lsqnonneg(C,d))
timeit(@() lsqnonneg_reg(C,d,Chi2Factor))

%%%%
%%%% EPGdecaycurve
%%%%
ETL = 32; flip_angle = 50.0; TE = 10e-3; T2 = 15e-3; T1 = 1000e-3; refcon = 180.0;

n_times = 1000;
call_epg = @() EPGdecaycurve(ETL, flip_angle, TE, T2, T1, refcon);
timeit(@() call_n_times(call_epg, n_times)) / n_times

%%%%
%%%% T2map_SEcorr
%%%%

n = 10;
M = 1e4 .* reshape(exp(-(1/6.0).*(1:32)) + exp(-(1/2.5).*(1:32)), 1,1,1,[]);
image = repmat(M, n, n, n, 1);
tic; [maps, distributions] = T2map_SEcorr(image, 'waitbar', 'no'); toc
