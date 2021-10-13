function [x] = fmg_lap(x, vSize)
%FMG_LAP Discrete Laplacian - central difference approximation
    
    if nargin < 2
        vSize = 1;
    end
    
    if isscalar(vSize)
        vSize = [vSize, vSize, vSize];
    end
    
    Mask = true(size(x));
    
    % *********************************************************************
    
    if isa(x, 'single')
        x = fmg_lap_mex_s(x, Mask, vSize);
    elseif isa(x, 'double')
        x = fmg_lap_mex_d(x, Mask, vSize);
    else
        error('x must be single or double.');
    end
    
end

% function [dx] = lap_conv(x, vSize)
%     
%     vSize = vSize .* vSize;
%     
%     h  = [1, -2, 1];
%     
%     hx = h  / vSize(1);
%     hy = h' / vSize(2);
%     hz = reshape(h, [1,1,length(h)]) / vSize(3);
%     
%     dx =      convn(x, hx, 'same');
%     dx = dx + convn(x, hy, 'same');
%     dx = dx + convn(x, hz, 'same');
%     
%     dx([1,end],:,:) = 0;
%     dx(:,[1,end],:) = 0;
%     dx(:,:,[1,end]) = 0;
%     
% end
