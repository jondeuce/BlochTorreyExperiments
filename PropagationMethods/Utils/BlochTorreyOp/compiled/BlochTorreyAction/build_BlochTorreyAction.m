%==========================================================================
% Periodic BlochTorreyAction Operator
%==========================================================================

if ispc
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        BlochTorreyAction_cd.c
    
    % complex double version - ctranspose operator
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        trans_BlochTorreyAction_cd.c
    
    % complex double version for non-constant diffusivity coefficient
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        BTActionVariableDiff_cd.c

    % complex double version for non-constant diffusivity with mask
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        BTActionVariableDiffNeumann_cd.c
    
    % complex double version for non-constant diffusivity with mask
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        BTActionVariableDiffNeumannBoolMask_cd.c
else
    % complex double version
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp BlochTorreyAction_cd.c
    
    % complex double version - ctranspose operator
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp trans_BlochTorreyAction_cd.c
    
    % complex double version for non-constant diffusivity coefficient
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp BTActionVariableDiff_cd.c

    % complex double version for non-constant diffusivity with mask
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp BTActionVariableDiffNeumann_cd.c
    
    % complex double version for non-constant diffusivity with mask
    mex CFLAGS="-fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp" ...
        COPTIMFLAGS="-O3 -flto -DNDEBUG" ...
        LDOPTIMFLAGS="-O3 -flto" ...
        -lgomp BTActionVariableDiffNeumannBoolMask_cd.c
end