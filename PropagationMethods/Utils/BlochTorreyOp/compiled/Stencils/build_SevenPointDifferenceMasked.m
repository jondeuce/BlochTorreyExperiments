%==========================================================================
% Periodic SevenPointDifference Operator
%==========================================================================

CALLING_DIRECTORY = pwd;
cd(fileparts(mfilename('fullpath')));

if ispc
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        SevenPointDifferenceMasked_cd.c
    
else
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp SevenPointDifferenceMasked_cd.c    
end

cd(CALLING_DIRECTORY)
clear CALLING_DIRECTORY
