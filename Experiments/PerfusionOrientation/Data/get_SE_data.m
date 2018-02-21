function [Alpha_Range, dR2_Data, EchoTime, VoxelSize, VoxelCenter, GridSize] = get_SE_data(Alpha_Range)
%[Alpha_Range, dR2_Data, EchoTime, VoxelSize, VoxelCenter, GridSize] = get_SE_data(Alpha_Range)

% ---- Load data from local storage ---- %
d = load('SEPWI_Data.mat');
d = d.SimParams;

Angles_Deg = 2.5:5:87.5;
if nargin < 1
    Alpha_Range = Angles_Deg;
end

[Angles_Intersect,~,Angle_Idx] = intersect(Alpha_Range,Angles_Deg);

if ~isequal(sort(Alpha_Range(:)), sort(Angles_Intersect(:)))
    angles_str = ['alpha = ', mat2str(Angles_Intersect(:).',4)];
    warning(['Requested alpha angles are not all present; ', ...
        'the following valid angles will be used:\n\n\t%s\n'], angles_str);
end
Alpha_Range = Angles_Intersect;

EchoTime = d.EchoTime; % [s]
dR2_Data = d.dR2_Peak(Angle_Idx); % [Hz]

VoxelSize   = [3000,3000,3000]; % [um]
VoxelCenter = [0,0,0]; % [um]
GridSize    = [512,512,512];

end

%--------------------------------------------------------------------------
% load SEPWI data directly
%--------------------------------------------------------------------------
%{
SEPWI_Results	=	load( [DataPath ...
    '/SEPWI/WhiteMatter_SEPWI_DTI_Data_Analysis_Results.mat']);
SEPWI_Data      =   struct(	...
    'type',         SEPWI_Results.type,           ...
    'TE',          	SEPWI_Results.TE,             ...
    'NumSubjects',  SEPWI_Results.subjNum,        ...
    'NumFrames',    SEPWI_Results.timeNum,        ...
    'PeakFrame',    SEPWI_Results.timePeakNum,    ...
    'Angles_Deg',   SEPWI_Results.Angles_Deg,     ...
    'Angles_Rad',	SEPWI_Results.Angles_Rad,     ...
    'time_frames',  SEPWI_Results.time_frames,    ...
    'time_s',       SEPWI_Results.time_s,         ...
    'dR2_Max',      SEPWI_Results.MAll,           ...
    'dR2_Min',      SEPWI_Results.mAll,           ...
    'S0',           SEPWI_Results.S0allSubj,      ...
    'SEPWI',        SEPWI_Results.SEPWIallSubj,   ...
    'BinCounts',    SEPWI_Results.NallSubj,       ...
    'dR2',          SEPWI_Results.meanR2allSubj	  ...
    );

SEPWI_Data	=   load('SEPWI_Data.mat');
SEPWI_Data	=   SEPWI_Data.SEPWI_Data;
%}

%--------------------------------------------------------------------------
% Get real data for matching with simulation
%--------------------------------------------------------------------------
%{
% Data is indexed as follows:
%   Dimension 1 is time, dim 2 is angle, and dim 3 is subject number
[TIME, ANGLE, SUBJECT]	=   deal(1,2,3);
WAvgData	=   @(x,KIND)	sum(x.*SEPWI_Data.BinCounts,KIND) ./	...
                            sum(SEPWI_Data.BinCounts,KIND);
AvgData     =   @(x,KIND)   mean(x,KIND);
MinData     =   @(x,KIND)	min(x,[],KIND);
MaxData     =   @(x,KIND)	max(x,[],KIND);

dR2_Avg     =   AvgData( SEPWI_Data.dR2, SUBJECT );
dR2_Peak	=   dR2_Avg( SEPWI_Data.PeakFrame, : );
Angles_Deg	=   double( SEPWI_Data.Angles_Deg );
Angles_Rad	=   double( SEPWI_Data.Angles_Rad );
EchoTime	=   double( SEPWI_Data.TE );
%}