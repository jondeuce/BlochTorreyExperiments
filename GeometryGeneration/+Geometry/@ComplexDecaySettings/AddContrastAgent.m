function [Gamma] = AddContrastAgent(GammaSettingsNoCA, GammaSettingsCA, Geom, Gamma)
% [Gamma] = AddContrastAgent(GammaSettingsNoCA, GammaSettingsCA, Geom, Gamma)
% Adds contrast agent to the existing complex decay array Gamma

% ---- Adjust Gamma to account for CA ---- %
if isempty(Geom.ArterialIndices) && isempty(Geom.VRSIndices)
    % Geometry is simply separated into two regions (blood-containing and
    % tissue), and calculation can be simplified as both susceptibility
    % and R2 maps are piecewise constant in each region
    Gamma = AddContrastAgent_NoArterialBlood_V2( GammaSettingsNoCA, GammaSettingsCA, Geom, Gamma );
else
    %There are more than two regions with different properties; Gamma must
    %be generically re-computed
    Gamma = GenericAddContrastAgent( GammaSettingsCA, Geom );
end

end

function Gamma = GenericAddContrastAgent( GammaSettingsCA, Geom )
% == Simplest way, but requires recomputation of dw (extra fft): == %

Gamma = CalculateComplexDecay( GammaSettingsCA, Geom );

end

function Gamma = AddContrastAgent_NoArterialBlood_V1( GammaSettingsNoCA, GammaSettingsCA, Geom, Gamma )
% == Version #1: Don't need to recompute dw; scales with dChi_blood == %

dChi_r = GammaSettingsCA.dChi_Blood / GammaSettingsNoCA.dChi_Blood;
dR2b_CA = GammaSettingsCA.dR2_Blood_CA;

R2 = real(Gamma) + (dR2b_CA .* Geom.VasculatureMap); % add CA contrib to R2 in blood
dw = imag(Gamma) .* dChi_r; % scale dw
Gamma = complex(R2,dw);

end

function Gamma = AddContrastAgent_NoArterialBlood_V2( GammaSettingsNoCA, GammaSettingsCA, Geom, Gamma )
% == Version #2: Can rewrite version #1 to not require temp variables: == %

dChi_r = GammaSettingsCA.dChi_Blood / GammaSettingsNoCA.dChi_Blood;
dR2b_CA = GammaSettingsCA.dR2_Blood_CA;
R2b = GammaSettingsNoCA.R2_Blood;
R2t = GammaSettingsNoCA.R2_Tissue;

Gamma = dChi_r .* Gamma;
Gamma = Gamma + ((1-dChi_r).*R2t);
Gamma = Gamma + ((1-dChi_r).*(R2b-R2t) + dR2b_CA) .* Geom.VasculatureMap;

end