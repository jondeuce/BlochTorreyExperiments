function y = SevenPointStencil_cd(x, kern, gsize4D, ndim, iters)
%SEVENPOINTSTENCIL_CD y = SevenPointStencil_cd(x, kern, gsize4D, ndim, iters)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_SevenPointStencil;
cd(currentpath);

y = SevenPointStencil_cd(x, kern, gsize4D, ndim, iters);

end
