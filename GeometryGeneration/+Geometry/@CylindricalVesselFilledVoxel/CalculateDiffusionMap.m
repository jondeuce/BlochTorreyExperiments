function [ D ] = CalculateDiffusionMap( G, D_Tissue, D_Blood, D_VRS )
%CALCULATEDIFFUSIONMAP [ D ] = CalculateDiffusionMap( G, D_Tissue, D_Blood, D_VRS )
% Computes the diffusion map associated with the geometry G with diffusion
% coefficients D_Tissue, D_Blood, and D_VRS in the tissue, blood, and VRS
% of the geometry, respectively.

if nargin < 4; D_VRS = []; end
if nargin < 3; D_Blood = []; end

if isempty(D_Blood) && isempty(D_VRS)
    % Constant coefficient isotropic diffusion
    D = D_Tissue;
else
    % Variable coefficient isotropic diffusion
    
    % Set blood diffusion coefficient
    if isempty(D_Blood)
        D = D_Tissue * ones(G.GridSize);
    else
        % D equals D_Blood where Vmap == 1, and D_Tissue where Vmap == 0
        D = (D_Blood - D_Tissue) .* G.VasculatureMap + D_Tissue;
    end
    
    % Set VRS diffusion coefficient
    if ~isempty(D_VRS) && ~isempty(G.VRSIndices)
        % set Virchow-Robin space R2 value
        D(G.VRSIndices) = D_VRS;
    end
end

end
