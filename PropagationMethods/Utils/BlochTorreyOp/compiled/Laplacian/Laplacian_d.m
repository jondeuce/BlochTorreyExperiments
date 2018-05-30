function y = Laplacian_d(x, h, gsize4D, ndim, iters)
%LAPLACIAN_D y = Laplacian_d(x, h, D, f, gsize4D, ndim, iters)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_Laplacian;
cd(currentpath);

y = Laplacian_d(x, h, gsize4D, ndim, iters);

end
