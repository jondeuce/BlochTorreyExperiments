%==========================================================================
% Periodic SevenPointStencil Operator
%==========================================================================

if ispc
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        SevenPointStencilMasked_cd.c
    
else
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp SevenPointStencilMasked_cd.c    
end