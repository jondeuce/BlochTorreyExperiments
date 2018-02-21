function x = padfastfft(x, padSize, forceSize, method)
% Pads matrix for Fourier Transform.
% 
%   x           - matrix to be padded
%   padSize     - pad size, The resulting image will have size: 
%                 size(x) = nextfastfft(size(x) + pad)
%   forceSize   - force padding with padSize. If odd, add the extra one
%                 before.
%   method      - padding method: 
%                 const (default: 0), 'circular', 'replicate', 'symmetric'
    
    if nargin < 4, method = 0; end
    
    [mSize0, padSize] = compatiblesizes(x,padSize);
    
    if nargin > 2 && forceSize
        x = padfastfft_(x, mSize0, padSize, method);
        return
    end
    
    mSize   = nextfastfft(mSize0 + padSize);
    padSize = mSize - mSize0;
    
    x = padfastfft_(x, padSize, method);
    
end

function [x] = padfastfft_(x, mSize, pad, method)
    
    odd0     = bitand(mSize, 1);
    odd1     = bitand(pad, 1);
    
    oddPre   = bitand(odd1, odd0);
    oddPost  = bitand(odd1, ~odd0);
    
    pad	     = floor(pad/2);
    
    if any(odd1)
        x = padarray(x, pad + oddPre, method, 'pre');
        x = padarray(x, pad + oddPost, method, 'post');
    else
        x = padarray(x, pad, method, 'both');
    end
    
end

function [mSize, padSize] = compatiblesizes(x,padSize)

    mSize = size(x);
    nx    = numel(mSize);
    np    = numel(padSize);
    
    mSize(nx+1:np)   = 1;
    padSize(np+1:nx) = 0;

end
