function [x] = fmg_restrict(x)

    %----------------------------------------------------------------------
    % Direct Methods
    %----------------------------------------------------------------------
%     x = x(1:2:end,1:2:end,1:2:end);
    
    %----------------------------------------------------------------------
    % Convolution Methods
    %----------------------------------------------------------------------
    h1  = [0.25; 0.5; 0.25];
    h2  = h1.';
    h3  = reshape(h1, [1,1,length(h1)]);
    h13 = bsxfun(@times,h1,h3);

    x  = padfastfft(x,[2,2,2],true,'circular');
    x  = convn(x, h13, 'valid');
    x  = convn(x, h2,  'valid');
%     x  = convn(x, hh, 'same');
%     x  = convn(x, h', 'same');

%     x  = x(1:2:end, 1:2:end, 1:2:end);
    x  = x(2:2:end, 2:2:end, 2:2:end);

end

%{
function [x] = injection(x)

x  = x(2:2:end, 2:2:end, 2:2:end);

end
%}
%{
    if nargin < 2
        g = true(size(x));
    else
        g = logical(g);
    end
    
    if isa(x, 'single')
        [x, g] = restrict_mex_s(x, g);
    elseif isa(x, 'double')
        [x, g] = restrict_mex_d(x, g);
    else
        [x, g] = restrict_mex_d(double(x), g);
    end
%}