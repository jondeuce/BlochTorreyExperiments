function [x] = lap(x, Mask, h)
%LAP [x] = lap(x, Mask, h)
% Discrete Laplacian - central difference approximation

    if nargin < 2 || isempty(Mask), Mask = true(size(x)); end
    if nargin < 3,  h = [1, 1, 1]; end
    
    if ~isa(Mask, 'logical'), Mask = logical(Mask); end
    
    x = lap_mex(x, Mask, h);
    
end

% function [dx] = lap_conv(x, h)
%     
%     h  = h .* h;
%     
%     h  = [1, -2, 1];
%     
%     hx = h  / h(1);
%     hy = h' / h(2);
%     hz = reshape(h, [1,1,length(h)]) / h(3);
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
