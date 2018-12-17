function y = BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters, isdiag)
%BLOCHTORREYACTION_CD y = BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters, isdiag)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_BlochTorreyAction;
cd(currentpath);

y = BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters, isdiag);

end
