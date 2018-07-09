function [ G_out ] = DilateMinorVessels( G, DilationFactor )
%DILATEMINORVESSELS Dilates the minor blood vessel radii such that the
%isotropic blood volume becomes DilationFactor * G.iBVF

G_out = G;
G_out.MinorDilation = DilationFactor;
G_out.MinorRadiusFactor = sqrt(DilationFactor);
G_out.isMinorDilated = true;

if nargin < 2
    DilationFactor = G.MinorDilation;
end

% Check if there are minor vessels to dilate
if G.Nminor == 0
    return
end

if abs(DilationFactor - 1.0) < 1e-8
    return
elseif DilationFactor < 0
    error('DilationFactor must be positive');
end

% Theoretically, scaling the radii by sqrt(DilationFactor) will scale the
% iBVF by DilationFactor, but we can do better by iteratively choosing the
% radii with fzero
InitialRadii = G_out.r;
InitialRadiusFactor = sqrt(DilationFactor);
fzero_opts = optimset('TolX',1e-5);%,'PlotFcns',{@optimplotx,@optimplotfval});
G_out.MinorRadiusFactor = fzero( @BVF_Error, InitialRadiusFactor, fzero_opts );

    function BVF_err = BVF_Error(RadiusScaleFactor)
        if abs(RadiusScaleFactor - 1.0) > 1e-8
            G_out.r = RadiusScaleFactor * InitialRadii;
            G_out.MinorRadiusFactor = RadiusScaleFactor;
            G_out = CalculateVasculatureMap(G_out);
        end
        BVF_err = G_out.Targets.iBVF * DilationFactor - G_out.iBVF;
    end

%Update properties which depend on the Vasculature Map
G_out = SetArteries(G_out);
G_out = SetVirchowRobinSpace(G_out);

end

