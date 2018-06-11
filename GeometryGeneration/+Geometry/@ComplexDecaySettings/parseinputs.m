function [ GammaSettings ] = parseinputs(GammaSettings, varargin)
%PARSEINPUTS [ GammaSettings ] = parseinputs(GammaSettings, varargin)

DefaultArgs   =   struct(...
    'Dimension',         3,              ... % Dimension of simulation
    'GyroMagRatio',      2.67515255e8,   ... % Gyromagnetic ratio [rad/(T*s)]
    'Angle_Deg',         90,             ... % Angle of B0 w.r.t. z-axis [deg]
    'B0',               -3.0,            ... % External magnetic field [T]
	'Hct',               0.44,           ... % Hematocrit fraction [fraction]
    'Y',                 0.61,           ... % Venous Blood Oxygenation [fraction]
    'CA',                0.0,            ... % Contrast Agent concentration [mM]
    'dR2_CA_per_mM',     5.2,            ... % Relaxation constant of the CA [Hz/mM]
    'dChi_CA_per_mM',    0.3393e-6,      ... % Susceptibility CA [[T/T]/mM]
    'Ya',                0.98            ... % Arterial Blood Oxygenation [fraction] Ref: Zhao et al., 2007, MRM, Oxygenation and hematocrit dependence of transverse relaxation rates of blood at 3T
    );

p = getParser(DefaultArgs);
parse(p,varargin{:});
args = p.Results;
args = appendDerivedQuantities(args);

for f = fieldnames(args).'
    fname = f{1};
    GammaSettings.(fname) = args.(fname);
end

end

function p = getParser(DefaultArgs)

p = inputParser;
for f = fields(DefaultArgs).'
    paramName = f{1};
    defaultVal = DefaultArgs.(f{1});
    addParameter(p,paramName,defaultVal)
end

end

function args = appendDerivedQuantities(args)

% Set the angle in radians as well
args.Angle_Rad = args.Angle_Deg * (pi/180);
    
% Get R2 default values based on current defaults
[   ...
    args.R2_Tissue,             ... % Total R2 in Tissue [ms]
    args.R2_Tissue_Base,        ... % Baseline R2 in Tissue [ms]
    args.R2_Blood,        ... % Total R2 in Venous Blood [ms]
    args.dR2_Blood_Oxy,   ... % Change in R2 in Venous Blood due to Oxygenation [ms]
    args.R2_ArterialBlood,      ... % Total R2 in Arterial Blood [ms]
    args.dR2_ArterialBlood_Oxy, ... % Change in R2 in Arterial Blood due to Oxygenation [ms]
    args.dR2_Blood_CA,          ... % Change in R2 in both Venous and Arterial Blood due to CA [ms]
    args.R2_CSF,                ... % CSF relaxation constant [Hz] 
    args.R2_VirchowRobin,       ... % Virchow-Robin space relaxation constant [Hz]
	] = ...
    R2_Model(args.B0, args.Hct, args.Y, args.Ya, args.CA, args.dR2_CA_per_mM);

% Get change in susceptibility (dChi) default values based on current defaults
[   ...
    args.dChi_Blood,       ... % Total dChi in Venous Blood [T/T]
    args.dChi_Blood_Oxy,   ... % dChi in Venous Blood due to oxygenation and Hct [T/T]
    args.dChi_ArterialBlood,     ... % Total dChi in Arterial Blood [T/T]
    args.dChi_ArterialBlood_Oxy, ... % dChi in Arterial Blood due to oxygenation and Hct [T/T]
    args.dChi_Blood_CA           ... % dChi in Blood due contrast agent [T/T]
    ] = ...
    dChi_Model(args.B0, args.Hct, args.Y, args.Ya, args.CA, args.dChi_CA_per_mM);

end

function [R2_Tissue, R2_Tissue_Base, R2_Blood, dR2_Blood_Oxy, ...
          R2_ArterialBlood, dR2_ArterialBlood_Oxy, dR2_Blood_CA, ...
          R2_CSF, R2_VirchowRobin] = ...
          R2_Model(B0, Hct, Yv, Ya, CA, dR2_CA_per_mM)
        
if B0 == -3.0
    % Relaxation constant in blood vs. tissue as a function of Hct and Y:
    % 	Zhao et al., Oxygenation and Hematocrit Dependence of Transverse
    %   Relaxation Rates of Blood at 3T (2007)
    if ~all(Hct == 0.21 | Hct == 0.44 | Hct == 0.57)
        error('Hct must be one of 0.21, 0.44, or 0.57 for R2(Hct,Y) model. See MRM 2007 Zhao et al., Oxygenation and hematocrit dependence of transverse relaxation rates of blood at 3T');
    end
    
    [A,B,C] = deal(0);
    if true
        % B coeff. == 0 Model (More physical)
        A(Hct == 0.21) = 8.2;  B(Hct == 0.21) = 0; C(Hct == 0.21) = 91.6;
        A(Hct == 0.44) = 11.0; B(Hct == 0.44) = 0; C(Hct == 0.44) = 125;
        A(Hct == 0.57) = 14.3; B(Hct == 0.57) = 0; C(Hct == 0.57) = 152;
    else
        % B coeff. ~= 0 Model (Less physical, better fit)
        A(Hct == 0.21) = 6.0;  B(Hct == 0.21) = 21.0; C(Hct == 0.21) = 94.3;
        A(Hct == 0.44) = 8.3;  B(Hct == 0.44) = 33.6; C(Hct == 0.44) = 71.9;
        A(Hct == 0.57) = 10.6; B(Hct == 0.57) = 39.3; C(Hct == 0.57) = 61.6;
    end
    
    dR2_Blood_Oxy   = A + B.*(1-Yv) + C.*(1-Yv).^2;
    dR2_ArterialBlood_Oxy = A + B.*(1-Ya) + C.*(1-Ya).^2;
    
    % T2 Value in WM @ 3 Tesla:
    %    Deistung et al. Susceptibility Weighted Imaging at Ultra High,
    %    Magnetic Field Strengths: Theoretical Considerations and
    %    Experimental Results, MRM 2008
    T2_Tissue_Base = 69; % +/- 3 [ms]
    R2_Tissue_Base = 1000/T2_Tissue_Base; % [ms] -> [Hz]
    
    %Spijkerman J, Petersen E, Hendrikse J, et al. T2 mapping of cerebrospinal fluid: 3T versus 7T.
    %Magnetic Resonance Materials in Physics, Biology and Medicine. Epub ahead of print 6 November 2017. DOI: 10.1007/s10334-017-0659-3.
    T2_CSF = 1790; % T2 of CSF at 3T [ms]; value corrected for partial volume effects; uncorrected value is 1672 ms
    R2_CSF = 1000/T2_CSF; % CSF relaxation constant [ms] -> [Hz] 
    R2_VirchowRobin = R2_CSF; % Virchow-Robin space relaxation constant [Hz]
    
elseif B0 == -7.0
    
    if ~all(abs(Hct-0.44) <= 0.01)
        error('Hct fraction must be within 0.01 of the nominal value of 0.44');
    end
    
    % Fit line through R2 vs. (1-Y)^2 data based on measurements from:
    %    Yacoub et al. Imaging Brain Function in Humans at 7 Tesla, MRM 2001
    Ydata = [0.38,0.39,0.59];
    T2data = [6.8,7.1,13.1];
    R2data = 1000./T2data;
    PP = polyfit((1-Ydata).^2,R2data,1);
    
    dR2_Blood_Oxy   = polyval(PP,(1-Yv).^2);
    dR2_ArterialBlood_Oxy = polyval(PP,(1-Ya).^2);
    
    % T2 Value in WM @ 7 Tesla:
    %    Deistung et al. Susceptibility Weighted Imaging at Ultra High,
    %    Magnetic Field Strengths: Theoretical Considerations and
    %    Experimental Results, MRM 2008
    
    T2_Tissue_Base = 45.9; % +/-1.9 [ms]
    R2_Tissue_Base = 1000/T2_Tissue_Base; % [ms] -> [Hz]
    
    %Spijkerman J, Petersen E, Hendrikse J, et al. T2 mapping of cerebrospinal fluid: 3T versus 7T.
    %Magnetic Resonance Materials in Physics, Biology and Medicine. Epub ahead of print 6 November 2017. DOI: 10.1007/s10334-017-0659-3.
    T2_CSF = 1010; % T2 of CSF at 7T [ms]; value corrected for partial volume effects; uncorrected value is 892 ms
    R2_CSF = 1000/T2_CSF; % CSF relaxation constant [ms] -> [Hz] 
    R2_VirchowRobin = R2_CSF; % Virchow-Robin space relaxation constant [Hz]
    
else
    error(['R2 relaxation times are only configured for B0 = -3T or -7T; ', ...
           'input B0 is %s'], num2str(B0,3));
end

% R2_Blood_Base = 31.1; % Typical value [ms]
dR2_Blood_CA = dR2_CA_per_mM * CA; % R2 change due to CA [ms]
R2_Blood = dR2_Blood_Oxy + dR2_Blood_CA; % Total R2 [ms]
R2_ArterialBlood = dR2_ArterialBlood_Oxy + dR2_Blood_CA; % Total R2 [ms]

R2_Tissue = R2_Tissue_Base; % R2 in tissue; no other contributions [ms]

end

function [dChi_Blood, dChi_Blood_Oxy, ...
          dChi_ArterialBlood, dChi_ArterialBlood_Oxy, dChi_Blood_CA] = ...
          dChi_Model(B0, Hct, Yv, Ya, CA, dChi_CA_per_mM)

% Susceptibility difference in blood vs. tissue including contrast agent
dChi_Blood_CA = CA * dChi_CA_per_mM;

% Susceptibilty of blood relative to tissue due to blood oxygenation and 
% hematocrit concentration is given by:
%   deltaChi_Blood_Tissue  :=   Hct * (1-Y) * 2.26e-6 [T/T]
dChi_Blood_Oxy = 2.26e-6 * Hct .* (1-Yv);
dChi_ArterialBlood_Oxy = 2.26e-6 * Hct .* (1-Ya);

% Susceptibility difference in blood vs. tissue including contrast agent as well
dChi_Blood = dChi_Blood_Oxy + dChi_Blood_CA;
dChi_ArterialBlood = dChi_ArterialBlood_Oxy + dChi_Blood_CA;

end
