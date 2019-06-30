function [Data] = miso_remake_sol(miso_initfun, miso_settings)
%MISO_REMAKE_SOL 

global sampledata

Data = miso_initfun();

Data.maxeval            = miso_settings{1};
Data.surrogate          = miso_settings{2};
Data.number_startpoints = miso_settings{3};
Data.init_design        = miso_settings{4};
Data.sampling           = miso_settings{5};
Data.own_design         = miso_settings{6};
Data.tol                = 0.001 * min(Data.xup - Data.xlow);
Data.S                  = sampledata(:, 1:Data.dim);
Data.m                  = size(sampledata, 1);
Data.Y                  = sampledata(:, Data.dim + 1);
Data.T                  = sampledata(:, Data.dim + 2);
[Data.fbest, best_idx]  = min(Data.Y);
Data.xbest              = sampledata(best_idx, 1:Data.dim);
Data.total_T            = sum(Data.T);

end