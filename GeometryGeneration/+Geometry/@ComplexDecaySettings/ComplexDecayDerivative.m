function [ dGamma ] = ComplexDecayDerivative(GammaSettings, Geom, Gamma, Variable, varargin)
%ComplexDecayDerivative

if nargin < 5; varargin = {}; end

switch upper(Variable)
    case 'CA'
        if isempty(Geom.ArterialIndices) && isempty(Geom.VRSIndices) && isempty(varargin)
            % No arteries or VRS, so can compute derivative exactly and
            % efficiently (no fft's) using only Gamma (no Gamma_0 needed)
            dGamma = CADerivative_NoArteries(GammaSettings_CA, Geom, Gamma, varargin{:});
        else
            % Gamma depends linearly on CA, so finite difference is exact,
            % both with and without arteries!
            % However, this requires an fft if Gamma_0 is not provided in
            % varargin, and this is the reason for the above branch.
            dGamma = CADerivative_Linear(GammaSettings, Geom, Gamma, varargin{:});
        end
end

end

function dGamma = CADerivative_Linear(GammaSettings_CA, Geom, Gamma_CA, varargin)
% This produces an exact derivative w.r.t. CA due to the fact that Gamma is
% an affine (not linear) function of CA, i.e.
%   Gamma = A + B*CA
% for some spatially dependent non-constant arrays A and B. We may
% interpret A as being Gamma_0, i.e. gamma with no contrast agent, and
% B as the derivative d(Gamma)/d(CA).
% 
% We can now easily solve for B given Gamma_CA and Gamma_0, and see that
%   d(Gamma)/d(CA) = (Gamma_CA - Gamma_0)/CA

% Get Gamma_0 from args if given, else calculate it
if isempty(varargin)
    GammaSettings_NoCA = GammaSettings_CA;
    GammaSettings_NoCA.CA = 0;
    Gamma_0 = CalculateComplexDecay( GammaSettings_NoCA, Geom );
else
    Gamma_0 = varargin{1};
end

% Calculate derivative with finite differences
dGamma = (Gamma_CA - Gamma_0)./(GammaSettings_CA.CA);

end

function dGamma = CADerivative_NoArteries(GammaSettings_CA, Geom, Gamma, varargin)

R2t = GammaSettings_CA.R2_Tissue;
R2b = GammaSettings_CA.R2_Blood;
dR2b_dCA = GammaSettings_CA.dR2_CA_per_mM;

dChib = GammaSettings_CA.dChi_Blood;
ddChi_dCA = GammaSettings_CA.dChi_CA_per_mM;

if isempty(Gamma)
    Gamma = CalculateComplexDecay(GammaSettings_CA, Geom);
end

% == Version #1: Simplest, but approx. twice as slow == %
r2 = (dR2b_dCA/(R2b-R2t)) * (real(Gamma)-R2t);
dw = (ddChi_dCA/dChib) * imag(Gamma);
dGamma = complex( r2, dw );

% == Version #2: Equivalent to above and faster, but uglier == %
% a = (dR2b_dCA/(R2b-R2t));
% b = (ddChi_dCA/dChib);
% dGamma = ((a-b) * real(Gamma)) - a*R2t;
% dGamma = dGamma + b * Gamma ;

end

function dGamma = CADerivative_WithArteries(GammaSettings_CA, Geom, Gamma, varargin)

%Due to separation of venous/arterial blood and convolution with dipole
%kernel, need to generically recompute dGamma with dGamma settings
dGammaSettings_dCA = GammaSettings_CA;
dGammaSettings_dCA.CA = 1;

dGammaSettings_dCA.dChi_Blood_CA = dGammaSettings_dCA.dChi_CA_per_mM;
dGammaSettings_dCA.dChi_Blood_Oxy = 0;
dGammaSettings_dCA.dChi_ArterialBlood_Oxy = 0;

dGammaSettings_dCA.dChi_Blood = dGammaSettings_dCA.dChi_CA_per_mM;
dGammaSettings_dCA.dChi_ArterialBlood = dGammaSettings_dCA.dChi_CA_per_mM;

dGammaSettings_dCA.dR2_Blood_CA = dGammaSettings_dCA.dR2_CA_per_mM;
dGammaSettings_dCA.dR2_Blood_Oxy = 0;
dGammaSettings_dCA.dR2_ArterialBlood_Oxy = 0;
dGammaSettings_dCA.R2_Tissue_Base = 0;

dGammaSettings_dCA.R2_Blood = dGammaSettings_dCA.dR2_CA_per_mM;
dGammaSettings_dCA.R2_Tissue = dGammaSettings_dCA.R2_Tissue_Base;
dGammaSettings_dCA.R2_ArterialBlood = dGammaSettings_dCA.dR2_CA_per_mM;

dGamma = CalculateComplexDecay( dGammaSettings_dCA, Geom );

end

% Testing for ComplexDecayDerivative
%{
gsetca = @(CA) Geometry.ComplexDecaySettings('Angle_Deg', alpha, 'B0', B0, 'CA', CA);
fgca = @(CA) CalculateComplexDecay(gsetca(CA), Geom);
fdgca = @(CA) ComplexDecayDerivative(gsetca(CA), Geom, fgca(CA), 'CA');

gca0 = fgca(0);
gca = fgca(CA);

dCA = CA*1e-1;
dgca_exact = fdgca(CA);
dgca_centered = (fgca(CA+dCA)-fgca(CA-dCA))/(2*dCA);
dgca_linear = (gca-gca0)/CA;

[ maxabs(dgca_exact-dgca_centered); maxabs(dgca_exact-dgca_linear) ]
%}