function [dx] = grad_adj(x, y, z, Mask, h)
%GRAD_ADJ [dx] = grad_adj(x, y, z, Mask, h)

    if nargin < 4 || isempty(Mask), Mask = true(size(x)); end
    if nargin < 5,  h = [1, 1, 1]; end
    
    if ~isa(Mask, 'logical'), Mask = logical(Mask); end
    
    dx = grad_adj_mex(x, y, z, Mask, h);
    
end

% function [dx] = conv_grad(x, y, z, h)
%     
%     dy = circshift(y, [1, 0, 0]);
%     dy = dy - y;
%     dy(1,:,:)   = -y(1,:,:);
%     dy(end,:,:) =  y(end-1,:,:); clear y
%     
%     dx = circshift(x, [0, 1, 0]);
%     dx = dx - x;
%     dx(:,1,:)   = -x(:,1,:);
%     dx(:,end,:) =  x(:,end-1,:); clear x
%     
%     dz = circshift(z, [0, 0, 1]);
%     dz = dz - z;
%     dz(:,:,1)   = -z(:,:,1);
%     dz(:,:,end) =  z(:,:,end-1); clear z
%     
%     if nargin < 4
%         dx = dx + dy + dz;
%     else
%         dx = dx./h(1) + dy./h(2) + dz./h(3);
%     end
%     
% end
