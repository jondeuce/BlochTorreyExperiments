function temperature = temperaturefastwbnds(optimValues,options)
%TEMPERATUREFASTWBNDS Updates the temperature vector for annealing process 
%   TEMPERATURE = TEMPERATUREFAST(optimValues,options) uses fast 
%   annealing by updating the current temperature based on the initial
%   temperature and the current annealing parameter k
%
%   OPTIMVALUES is a structure containing the following information:
%              x: current point 
%           fval: function value at x
%          bestx: best point found so far
%       bestfval: function value at bestx
%    temperature: current temperature
%      iteration: current iteration 
%             t0: start time
%              k: annealing parameter
%
%   OPTIONS: options structure created by using SAOPTIMSET.
%
%   Example:
%    Create an options structure using TEMPERATUREFAST as the annealing
%    function
%    options = saoptimset('TemperatureFcn' ,@temperaturefast);

%   jd: parameter list:
%   temp(1) = CA_Concentration      % contrast agent concentration (mM)
%   temp(2) = Total_BVF             % blood volume fraction of voxel
%   temp(3) = MinorVessel_RelBVF	% fraction of Total_BVF contained in minor vasculature

temperature = options.InitialTemperature./optimValues.k;

% All parameters are normalized to be in the range [0,1]; choose desired
% accuracy based this range
r(1) = -2; % CA_Concentration round to the nearest hundredth
r(2) = -2; % Total_BVF round to the nearest hundredth
r(3) = -2; % MinorVessel_RelBVF round to the nearest hundredth

temperature(1)=roundn(temperature(1),r(1));
temperature(2)=roundn(temperature(2),r(2));
temperature(3)=roundn(temperature(3),r(3));

%check if this is the same as the previous one
while options.InitialTemperature==temperature
    temperature=temperautre./optimValues.k;
    temperature(1)=roundn(temperature(1),r(1));
    temperature(2)=roundn(temperature(2),r(2));
    temperature(3)=roundn(temperature(3),r(3));
end





