function y = sor_diffuse_cd(b, x, w, h, D, f, s, c, iter, dir)
%SOR_DIFFUSE_CD y = sor_diffuse_cd(b, x, w, h, D, f, s, c, iter, dir)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
build_sor_diffuse;
cd(currentpath);

y = sor_diffuse_cd(b, x, w, h, D, f, s, c, iter, dir);

end

