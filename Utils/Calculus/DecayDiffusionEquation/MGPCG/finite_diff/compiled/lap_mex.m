function x = lap_mex(x, Mask, h)
%LAP_MEX x = lap_mex(x, Mask, h)

currentpath = cd;
cd(fileparts(mfilename('fullpath')));
mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
    COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
    LDOPTIMFLAGS="-O3 -flto" ...
    -lgomp lap_mex.c
cd(currentpath);

x = lap_mex(x, Mask, h);

end

