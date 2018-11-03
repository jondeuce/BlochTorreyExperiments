function [ output_args ] = splinefillingtest( Results, type )
%SPLINEFILLINGTEST

if nargin < 2; type = 'plot'; end

switch upper(type)
    case 'PLOT'
        % NumTEs = 10, NumAngles = 10 (102 steps/angle; 6.8e-3 max error)
        TEs1    =  1e-3 * vec([0     5    15    20    30    50    60    75    90   120]);
        Angles1 = deg2rad(vec([0     5    15    25    45    60    70    75    85    90]));
        
        % NumTEs = 11, NumAngles = 12 (107 steps/angle; 3.1e-3 max error)
        TEs1    =  1e-3 * vec([0     5    10    15    30    35    40    60    75    95   120]);
        Angles1 = deg2rad(vec([0     5    10    15    20    30    40    45    55    75    85    90]));
        
        % NumTEs = 10, NumAngles = 11 (84 steps/angle; 14.3e-3 max error)
        TEs1    =  1e-3 * vec([0     5    10    15    20    30    35    50    90   120]);
        Angles1 = deg2rad(vec([0    10    20    30    35    40    50    55    60    80    90]));
        
        % NumTEs = 12, NumAngles = 12 (118 steps/angle; 2.7e-3 max error)
        TEs1    =  1e-3 * vec([0     5    10    15    20    30    35    50    65    80   105   120]);
        Angles1 = deg2rad(vec([0     5    15    25    30    35    40    45    60    70    85    90]));
        
        % close all force
        plot(Results);
        InterpBOLDs = splinefill(TEs1, Angles1, Results, true);
        
    case 'FINDBEST'
        NumTEs = 12;
        NumAngles = 12;
        [TEsBest, AnglesBest, ErrorBest, NumStepsBest] = SE_find_best(NumTEs, NumAngles, Results);
end

end

function InterpBOLDs = splinefill(TEs1, Angles1, Results, plot)

if nargin < 4; plot = false; end

[AllTEs1, AllAngles1, AllBOLDsCell] = getBOLDSignals(Results);
[AllTEs, AllAngles] = meshgrid(AllTEs1(:), AllAngles1(:));
AllBOLDs = AllBOLDsCell{1}.';
AllBOLDs = AllBOLDs/max(AllBOLDs(:));

dTE = mean(diff(AllTEs1(:)));
dAngle = mean(diff(AllAngles1(:)));
idx_TEs1 = round(TEs1/dTE + 1);
idx_Angles1 = round(Angles1/dAngle + 1);

[TEs, Angles] = meshgrid(TEs1, Angles1);
BOLDs = AllBOLDs(idx_Angles1,idx_TEs1);

InterpBOLDs = interp2(TEs, Angles, BOLDs, AllTEs, AllAngles, 'spline');

if plot
    figure, surf(1000*AllTEs, 180/pi*AllAngles, AllBOLDs); xlabel('time [ms]'); ylabel('angles [deg]'); zlabel('original'); zlim([0,1]);
    figure, surf(1000*AllTEs, 180/pi*AllAngles, InterpBOLDs); xlabel('time [ms]'); ylabel('angles [deg]'); zlabel('interp'); zlim([0,1]);
    figure, surf(1000*AllTEs, 180/pi*AllAngles, (AllBOLDs - InterpBOLDs)./(AllBOLDs + 1e-12)); xlabel('time [ms]'); ylabel('angles [deg]'); zlabel('rel diff'); zlim([-3,3]*1e-3);
end

end

function [TEsBest, AnglesBest, ErrorBest, NumStepsBest] = SE_find_best(NumTEs, NumAngles, Results)

[AllTEs1, AllAngles1, AllBOLDsCell] = getBOLDSignals(Results);
AllTEs1 = AllTEs1(:);
AllAngles1 = AllAngles1(:);
AllBOLDs = AllBOLDsCell{1}.';
AllBOLDs = AllBOLDs/max(AllBOLDs(:));
dTE = mean(diff(AllTEs1(:)));
dAngle = mean(diff(AllAngles1(:)));

[ErrorBest, NumStepsBest] = deal(Inf);

stall_iter = 0;
STALL_ITER_MAX = 5000;

while true
    idx_TEs1 = sort([1, 1 + randperm(length(AllTEs1)-2, NumTEs-2), length(AllTEs1)]);
    idx_Angles1 = sort([1, 1 + randperm(length(AllAngles1)-2, NumAngles-2), length(AllAngles1)]);
    
    TEs1 = AllTEs1(idx_TEs1);
    Angles1 = AllAngles1(idx_Angles1);
    InterpBOLDs = splinefill(TEs1, Angles1, Results);
    
    Error = max(abs(vec( (AllBOLDs - InterpBOLDs)./(AllBOLDs + 1e-12) )));
    NumSteps = SE_num_timesteps(round(TEs1(2:end)/dTE), 2);
    
    if Error <= ErrorBest && NumSteps <= NumStepsBest
        TEsBest = TEs1;
        AnglesBest = Angles1;
        ErrorBest = Error;
        NumStepsBest = NumSteps;
        stall_iter = 0;
    else
        stall_iter = stall_iter + 1;
        if stall_iter >= STALL_ITER_MAX
            disp(round(TEsBest.'*1000))
            disp(round(rad2deg(AnglesBest.')))
            disp(1000*ErrorBest)
            disp(NumStepsBest)
            break
        end
    end
end


end

function s = SE_num_timesteps(N,m)

if N(1) == 0; N = N(2:end); end
if N(1) == 1; s = 0; else s = m/2; end

for n = N(:).'
    s = s + m/2 + n*m/2;
end

end

function s = GRE_num_timesteps(N,m)

s = N(end)*m;

end

