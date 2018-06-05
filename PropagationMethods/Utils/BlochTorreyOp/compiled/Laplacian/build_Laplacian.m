%==========================================================================
% Periodic Laplacian Operator
%==========================================================================

if ispc
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        Laplacian_cd.c
    
    % real double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        Laplacian_d.c
    
    % complex single version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        Laplacian_cs.c
    
else
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp Laplacian_cd.c
    
    % real double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp Laplacian_d.c
    
    % complex single version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp Laplacian_cs.c
    
end