%==========================================================================
% Periodic SevenPointStencil Operator
%==========================================================================

if ispc
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        SevenPointStencil_cd.c
    
%     % real double version
%     mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
%         COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
%         LDOPTIMFLAGS="-O3 -flto" ...
%         SevenPointStencil_d.c
    
%     % complex single version
%     mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
%         COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
%         LDOPTIMFLAGS="-O3 -flto" ...
%         SevenPointStencil_cs.c
    
else
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp SevenPointStencil_cd.c
    
%     % real double version
%     mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
%         COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
%         LDOPTIMFLAGS="-O3 -flto" ...
%         -lgomp SevenPointStencil_d.c
    
%     % complex single version
%     mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
%         COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
%         LDOPTIMFLAGS="-O3 -flto" ...
%         -lgomp SevenPointStencil_cs.c
    
end