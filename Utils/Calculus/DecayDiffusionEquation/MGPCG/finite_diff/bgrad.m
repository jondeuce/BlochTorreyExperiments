function [dx, dy, dz] = bgrad(x, Mask, h)
%BGRAD [dx, dy, dz] = bgrad(x, Mask, h)
% First order backward finite difference gradient with Dirichlet BC.

    if nargin < 2 || isempty(Mask), Mask = true(size(x)); end
    if nargin < 3,  h = [1, 1, 1]; end
    
    if ~isa(Mask, 'logical'), Mask = logical(Mask); end
    
    [dx, dy, dz] = bgrad_mex(x, Mask, h);

end

% function [dx,dy,dz] = conv_grad(img, h)
%     
%     dy = circshift(img, [1, 0, 0]);
%     dy = img - dy;
%     dy(1,:,:)   =  0;
%     
%     dx = circshift(img, [0, 1, 0]);
%     dx = img - dx;
%     dx(:,1,:)   =  0;
%     
%     dz = circshift(img, [0, 0, 1]);
%     dz = img - dz;
%     dz(:,:,1)   =  0;
%     
%     if nargin > 1
%         dx = dx ./ h(1);
%         dy = dy ./ h(2);
%         dz = dz ./ h(3);
%     end
%     
% end
