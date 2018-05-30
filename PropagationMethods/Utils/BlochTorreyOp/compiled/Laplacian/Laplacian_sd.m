function y = Laplacian_sd(x, h, gsize4D, ndim, iters)
%LAPLACIAN_sd y = Laplacian_sd(x, h, D, f, gsize4D, ndim, iters)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_Laplacian;
cd(currentpath);

y = Laplacian_sd(x, h, gsize4D, ndim, iters);

end
