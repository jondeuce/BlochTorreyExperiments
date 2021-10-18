function [dx, dy, dz] = grad_mex(x, Mask, h)
%GRAD_MEX [dx, dy, dz] = grad_mex(x, Mask, h)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
    COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
    LDOPTIMFLAGS="-O3 -flto" ...
    -lgomp grad_mex.c
cd(currentpath);

[dx, dy, dz] = grad_mex(x, Mask, h);

end

