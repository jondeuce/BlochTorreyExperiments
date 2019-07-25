function [ G_out ] = DilateMajorVessels( G, DilationFactor )
%DILATEMAJORVESSELS Dilates the major blood vessel radii such that the
%isotropic blood volume becomes DilationFactor * G.aBVF

G_out = G;

if nargin < 2
    DilationFactor = G.MajorDilation;
end

if abs(DilationFactor - 1.0) < 1e-8
    return
elseif DilationFactor < 0
    error('DilationFactor must be positive');
end

G_out.MajorDilation = DilationFactor;
G_out.MajorRadiusFactor = sqrt(DilationFactor);
G_out.isMajorDilated = true;

% Theoretically, scaling the radii by sqrt(DilationFactor) will scale the
% aBVF by DilationFactor, but we can do better by iteratively choosing the
% radii with fzero
InitialRadii = G_out.r0;
InitialRadiusFactor = sqrt(DilationFactor);
fzero_opts = optimset('TolX',1e-5); %,'PlotFcns',{@optimplotx,@optimplotfval});
G_out.MajorRadiusFactor = fzero( @BVF_Error, InitialRadiusFactor, fzero_opts );

    function BVF_err = BVF_Error(RadiusScaleFactor)
        if abs(RadiusScaleFactor - 1.0) > 1e-8
            G_out = SetRmajor(G_out, RadiusScaleFactor * InitialRadii);
            G_out.MajorRadiusFactor = RadiusScaleFactor;
            G_out = CalculateVasculatureMap(G_out);
        end
        BVF_err = G_out.Targets.aBVF * DilationFactor - G_out.aBVF;
    end

%Update properties which depend on the Vasculature Map
G_out = Uncompress(G_out);

end
