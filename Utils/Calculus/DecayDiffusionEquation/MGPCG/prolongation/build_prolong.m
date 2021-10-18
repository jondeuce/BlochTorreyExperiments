% Successive over-relaxation smoothing for the transmag-diffusion problem

CALLING_DIRECTORY = pwd;
cd(fileparts(mfilename('fullpath')));

if ispc
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        prolong_cd.c
else
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp prolong_cd.c
end

cd(CALLING_DIRECTORY)
clear CALLING_DIRECTORY
