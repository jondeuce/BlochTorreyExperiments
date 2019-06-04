TOOLS FOR CALCULATING MYELIN WATER FRACTION

Edit history:
-Swap parfor -> for loop
-Remove various waitbars/print statements

Example usage:

%addpath /data/workgroup/matlab/testing_functions/
%addpath /data/ubcitar/aweber/MWI/SE_corr

[MWImaps, MWIdist] = T2map_SEcorr(mag); % mag is an [row x col x slice x echos] 4D array
[MWIpart] = T2part_SEcorr(MWIdist, 'spwin', [0.014, 0.04]); % small pool window

MWIpart.sfr the myelin water fraction:
s - short --> myelin
m - medium --> intra/extra water
fr - fraction
gm - geometric mean