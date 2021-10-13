function [dx, dy, dz] = bgrad_mex(x, Mask, h)
%BGRAD_MEX Summary of this function goes here
%   Detailed explanation goes here

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
    COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
    LDOPTIMFLAGS="-O3 -flto" ...
    -lgomp bgrad_mex.c
cd(currentpath);

[dx, dy, dz] = bgrad_mex(x, Mask, h);

end

