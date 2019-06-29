%==========================================================================
% Periodic SevenPointStencil Operator
%==========================================================================

CALLING_DIRECTORY = pwd;
cd(fileparts(mfilename('fullpath')));

if ispc
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        SevenPointStencilMasked_cd.c
    
else
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp SevenPointStencilMasked_cd.c    
end

cd(CALLING_DIRECTORY)
clear CALLING_DIRECTORY
