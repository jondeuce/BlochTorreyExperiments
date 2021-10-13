%==========================================================================
% periodic decay diffusion operator
%==========================================================================

CALLING_DIRECTORY = pwd;
cd(fileparts(mfilename('fullpath')));

if ispc
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -Ofast -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-Ofast -flto" ...
        fmg_diffuse_cd.c
else
    
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-march=native -msse2 -msse3 -Ofast -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-Ofast -flto" ...
        -lgomp fmg_diffuse_cd.c
end

% %==========================================================================
% % Gauss-Siedel for periodic decay diffusion operator
% %==========================================================================
%
% % complex double version
% mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
%     COPTIMFLAGS="-march=native -msse2 -msse3 -Ofast -flto -DNDEBUG" ...
%     LDOPTIMFLAGS="-Ofast -flto" ...
%     -lgomp fmg_relax_gs_diffuse_cd.c

cd(CALLING_DIRECTORY)
clear CALLING_DIRECTORY
