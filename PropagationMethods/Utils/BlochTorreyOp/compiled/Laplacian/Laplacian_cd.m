function y = Laplacian_cd(x, h, gsize4D, ndim, iters)
%LAPLACIAN_CD y = Laplacian_cd(x, h, gsize4D, ndim, iters)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_Laplacian;
cd(currentpath);

y = Laplacian_cd(x, h, gsize4D, ndim, iters);

end
