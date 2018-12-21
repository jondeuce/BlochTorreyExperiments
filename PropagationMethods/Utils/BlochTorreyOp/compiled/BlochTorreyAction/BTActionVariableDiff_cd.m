function y = BTActionVariableDiff_cd(x, h, D, f, gsize4D, ndim, iters, isdiag)
%BTACTIONVARIABLEDIFF_CD y = BTActionVariableDiff_cd(x, h, D, f, gsize4D, ndim, iters, isdiag)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_BlochTorreyAction;
cd(currentpath);

y = BTActionVariableDiff_cd(x, h, D, f, gsize4D, ndim, iters, isdiag);

end
