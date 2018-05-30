function y = trans_BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters)
%TRANS_BLOCHTORREYACTION_CD y = trans_BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_BlochTorreyAction;
cd(currentpath);

y = trans_BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters);

end
