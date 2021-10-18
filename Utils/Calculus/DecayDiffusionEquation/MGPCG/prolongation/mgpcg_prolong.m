function [y] = mgpcg_prolong(x)

    ySize = 2*size(x);
    y     = zeros(ySize,'like',x);
    
    y(1:2:end, 1:2:end, 1:2:end) = x;
    
    %----------------------------------------------------------------------
    % Direct Methods
    %----------------------------------------------------------------------
%     y(2:2:end,:,:) = 0.5*(y(1:2:end,:,:) + y([end-1,1:2:end-2],:,:));
%     y(:,2:2:end,:) = 0.5*(y(:,1:2:end,:) + y(:,[end-1,1:2:end-2],:));
%     y(:,:,2:2:end) = 0.5*(y(:,:,1:2:end) + y(:,:,[end-1,1:2:end-2]));
    
    %----------------------------------------------------------------------
    % Convolution Methods
    %----------------------------------------------------------------------
    h1   = [0.5; 1; 0.5];
    [h2,h3] = deal(h1.',reshape(h1,1,1,[]));
    h13  = bsxfun(@times,h1,h3);
    
%     y  = padfastfft(y,[2,2,2],true,'circular');
%     y  = convn(y, h13, 'valid');
%     y  = convn(y, h2,  'valid');
    
    y = imfilter(y,h13,'circular','same');
    y = imfilter(y,h2, 'circular','same');
    
%     y = conv_even_per(y,h1,1);
%     y = conv_even_per(y,h1,2);
%     y = conv_even_per(y,h1,3);
    
%     h123 = bsxfun(@times,h2,h13);
%     H = padfastfft(h123,ySize-size(h123),true,0);
%     H = fftn(ifftshift(H));
%     y = ifftn(fftn(y).*H);
    
end

%{
    if nargin < 2
        g = true(size(x));
    else
        g = logical(g);
    end
    
    if isa(x, 'single')
        x = prolong_mex_s(x, g);
    elseif isa(x, 'double')
        x = prolong_mex_d(x, g);
    else
        x = prolong_mex_d(double(x), g);
    end
%}