function [Alpha_Range, dR2_Data, EchoTime, VoxelSize, VoxelCenter, GridSize, BinCounts] = get_GRE_data(Alpha_Range)
%get_GRE_data

% ---- Load data from local storage ---- %
d = load('GREPWI_Data.mat');
d = d.GREPWI_Data;

Angles_Deg = d.Angles_Deg;
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

EchoTime = d.TE; % [s]
dR2_Data = d.dR2_Peak(Angle_Idx); % [Hz]

VoxelSize = [1750,1750,4000]; % [um]
VoxelCenter = VoxelSize/2; % [um]
GridSize = [350,350,800];

BinCounts = sum(d.BinCounts,3);
BinCounts = BinCounts(Angle_Idx);

end