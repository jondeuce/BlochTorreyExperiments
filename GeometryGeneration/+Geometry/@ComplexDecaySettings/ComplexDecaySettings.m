classdef ComplexDecaySettings
    %COMPLEXDECAYSETTINGS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (GetAccess = public, SetAccess = private)
        Angle_Deg
        Angle_Rad
        Dimension
        B0
        CA
        dChi_Blood
        dChi_ArterialBlood
        dChi_Blood_CA
        dChi_Blood_Oxy
        dChi_ArterialBlood_Oxy
        dChi_CA_per_mM
        dR2_Blood_CA
        dR2_Blood_Oxy
        dR2_ArterialBlood_Oxy
        dR2_CA_per_mM
        GyroMagRatio
        Hct
        R2_Blood
        R2_ArterialBlood
        R2_Tissue
        R2_Tissue_Base
        R2_VirchowRobin
        R2_CSF
        Y
        Ya
    end
    
    methods
        
        function GammaSettings = ComplexDecaySettings(varargin)
            
            GammaSettings = Geometry.ComplexDecaySettings.parseinputs( ...
                GammaSettings, varargin{:});
            
        end
        
    end
    
    methods (Static = true)
        [ GammaSettings ] = parseinputs(GammaSettings, varargin)
    end
    
end

