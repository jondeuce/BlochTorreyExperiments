function x = grad_adj_mex(x, y, z, Mask, h)
%GRAD_ADJ_MEX Summary of this function goes here
%   Detailed explanation goes here

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
    COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
    LDOPTIMFLAGS="-O3 -flto" ...
    -lgomp grad_adj_mex.c
cd(currentpath);

x = grad_adj_mex(x, y, z, Mask, h);

end

