function [dx, dy, dz] = grad(x, Mask, h)
%GRAD [dx, dy, dz] = grad(x, Mask, h)
% First order forward finite difference gradient with Dirichlet BC.

    if nargin < 2 || isempty(Mask), Mask = true(size(x)); end
    if nargin < 3,  h = [1, 1, 1]; end
    
    if ~isa(Mask, 'logical'), Mask = logical(Mask); end
    
    [dx, dy, dz] = grad_mex(x, Mask, h);

end

% function [dx,dy,dz] = conv_grad(img, h)
%     
%     dy = circshift(img, [-1, 0, 0]);
%     dy = dy - img;
%     dy(end,:,:) = 0;
%     
%     dx = circshift(img, [0, -1, 0]);
%     dx = dx - img;
%     dx(:,end,:) = 0;
%     
%     dz = circshift(img, [0, 0, -1]);
%     dz = dz - img;
%     dz(:,:,end) = 0;
%     
%     if nargin > 1
%         dx = dx / h(1);
%         dy = dy / h(2);
%         dz = dz / h(3);
%     end
%     
% end
