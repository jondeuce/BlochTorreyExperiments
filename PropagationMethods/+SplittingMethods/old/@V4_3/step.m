function y = step(V, y, h, D, Gamma)

if nargin < 3; h = []; end
if nargin < 4; D = []; end
if nargin < 5; Gamma = []; end

newTimeStep = ~isempty(h);
newGamma    = ~isempty(Gamma);
newDcoeff   = ~isempty(D);

if newTimeStep || newDcoeff
    % New time step or Dcoeff; recompute kernels
    if V.allowPreCompKernels
        V = precomputeKernels(V,h,D);
    else
        V = computeKernels(V,h,D);
    end
end

if (newTimeStep || newGamma) && V.allowPreCompDecays
    % New Time Step or Gamma; recompute decays
    V = precomputeDecays(V,h,Gamma);
end

% Note: Gaussian Kernel class handles it's own precomputation
if isempty(V.E1) || isempty(V.E2)
    
    y = exp( (-V.b1 * h) .* Gamma ) .* x0;
    y = conv( V.K1, y );
    y = exp( (-V.b2 * h) .* Gamma ) .* y;
    y = conv( V.K2, y );
    y = exp( (-V.b2 * h) .* Gamma ) .* y;
    y = conv( V.K1, y );
    y = exp( (-V.b1 * h) .* Gamma ) .* y;
    
else
    
    y = V.E1 .* x0;
    y = conv( V.K1, y );
    y = V.E2 .* y;
    y = conv( V.K2, y );
    y = V.E2 .* y;
    y = conv( V.K1, y );
    y = V.E1 .* y;
    
end

end
