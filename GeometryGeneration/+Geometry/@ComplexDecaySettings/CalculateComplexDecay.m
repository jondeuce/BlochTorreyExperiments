function [ varargout ] = CalculateComplexDecay(GammaSettings, Geom, OutputType)
%CALCULATECOMPLEXDECAY

if nargin < 3
    OutputType = 'Gamma';
end

switch upper(OutputType)
    case 'DOMEGA'
        varargout{1} = get_dOmega(GammaSettings, Geom);
    case 'R2'
        varargout{1} = get_R2(GammaSettings, Geom);
    otherwise
        % Calculate both
        r2 = get_R2(GammaSettings, Geom);
        dw = get_dOmega(GammaSettings, Geom);
        if nargout == 2
            varargout{1} = r2;
            varargout{2} = dw;
        else
            varargout{1} = complex(r2,dw);
        end
end

end

function dw = get_dOmega(GammaSettings, Geom)

gyro  = GammaSettings.GyroMagRatio;
B0    = GammaSettings.B0;
dChiV = GammaSettings.dChi_Blood; % venous blood dChi
dChiA = GammaSettings.dChi_ArterialBlood; % arterial blood dChi

alpha = GammaSettings.Angle_Rad;
BDir  = [sin(alpha); 0; cos(alpha)];

% tmp here is just dChi scaled by gyro*B0 to save a multiplication later
tmp   = (gyro * B0 * dChiV) .* Geom.VasculatureMap;
if ~isempty(Geom.ArterialIndices)
    tmp(Geom.ArterialIndices) = tmp(Geom.ArterialIndices) + (gyro * B0 * (dChiA - dChiV));
end

% convolve the (scaled) dChi map with the dipole kernel
d_kspace = Geometry.dipole(Geom.GridSize, Geom.VoxelSize, BDir);
tmp = fftn(tmp); % fft("scaled_dChi")
tmp = tmp .* d_kspace; % fft("scaled_dChi") .* fft_dipole (mult. in kspace)
clear d_kspace

dw = real(ifftn(tmp)); % ifft back to complete the convolution

end

function r2 = get_R2(GammaSettings, Geom)

R2_vb = GammaSettings.R2_Blood;
R2_ab = GammaSettings.R2_ArterialBlood;
R2t = GammaSettings.R2_Tissue;

% r2 equals R2_Blood where Vmap == 1, and R2_Tissue where Vmap == 0
r2 = (R2_vb - R2t) .* Geom.VasculatureMap + R2t;

% add in arterial blood effects, if present
if ~isempty(Geom.ArterialIndices)
    r2(Geom.ArterialIndices) = r2(Geom.ArterialIndices) + (R2_ab - R2_vb);
end

end
