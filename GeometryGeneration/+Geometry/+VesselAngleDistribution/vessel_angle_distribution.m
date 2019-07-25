function [ alpha_vessel ] = vessel_angle_distribution( alpha_fibre )
%VESSEL_ANGLE_DISTRIBUTION Outputs the effective vessel angle for a given
%fibre angle. See `vessel_angle_distribution.jl`.

data = load([fileparts(mfilename('fullpath')), '/2019-07-02-T-17-32-56-511.angledistribution.mat']);
pp = splinefit(data.FibreAngle, data.EffectiveAngle, vec(0:5:90));
alpha_vessel = ppval(pp, alpha_fibre);

end

