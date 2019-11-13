n = 10;
M = 1e4 .* reshape(exp(-(1/6.0).*(1:32)) + exp(-(1/2.5).*(1:32)), 1, 1, 1, []);
image = repmat(M, n, n, n, 1);

profile clear
profile on
[maps, distributions] = T2map_SEcorr(image, 'waitbar', 'no');
profsave(profile('info'), sprintf('tmp_prof_results/profile_results_%d', n))
