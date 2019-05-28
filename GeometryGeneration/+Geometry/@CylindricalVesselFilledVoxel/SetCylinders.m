function [ G ] = SetCylinders( G, p0, r0, vz0, p, r, vz, NumMajorArteries, MinorArterialFrac )
%SETCYLINDERS Sets the major cylinders of G to those specified by p0, r0,
% and vz0, and the minor cylinders of G to those specified by p, r, vz.
% Also, updates all other necessary properties, such as the vasculature map
% stored in G.

if nargin < 9; MinorArterialFrac = 0.0; end
if nargin < 8; NumMajorArteries = 0; end

t_StartTime = tic;

%==========================================================================
% Calculate Vascular Map from Major/Minor Cylinders
%==========================================================================

G.Nmajor = size(p0,2);
G.Nminor = size(p,2);
G.Rmajor = r0;
G.Rminor_mu = NaN;
G.Rminor_sig = NaN;
for f = fields(G.Targets).'
    G.Targets.(f{1}) = NaN;
end

G.N = G.Nmajor + G.Nminor;
G.P = [p0, p];
G.R = [r0, r];
G.Vz = [vz0, vz];
G = NormalizeCylinderVecs(G);

G = CalculateVasculatureMap( G );
G.BVF = sum(G.VasculatureMap(:))/prod(G.GridSize);
G.aBVF = sum(G.VasculatureMap(cat(1,G.idx0{:})))/prod(G.GridSize);
G.iBVF = G.BVF - G.aBVF;
G.iRBVF = G.iBVF/G.BVF;
G.aRBVF = G.aBVF/G.BVF;

%==========================================================================
% Add arteries
%==========================================================================

t_SetArteries = tic;

G.NumMajorArteries = NumMajorArteries;
G.MinorArterialFrac = MinorArterialFrac;
G = SetArteries(G);

t_SetArteries = toc(t_SetArteries);
if G.Verbose; display_toc_time( t_SetArteries, 'setting arteries', 0 ); end

%==========================================================================
% Show Resulting Accuracy Info
%==========================================================================

if G.Verbose; ShowBVFResults(G); end

%==========================================================================
% Return Geometry object, recording timing info
%==========================================================================

t_TotalTime = toc(t_StartTime);
if G.Verbose; display_toc_time( t_TotalTime, 'total cylinder construction time', 0 ); end

end

