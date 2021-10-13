function [ y ] = conv_even_per( x, g, dim )
%CONV_EVEN_PER Convolves the 2D or 3D complex array x with the kernel g.
% The kernel g must be an array with only one non-singleton dimension. The
% kernel g must have odd length and is assumed to be even about the center.

    if isscalar(g)
        y = g * x;
        return
    end

    if ~isreal(g)
        error('Kernel g must be real.');
    end
    
    if sum(size(g)~=1) ~= 1
        error('The kernel g must only have one non-singleton dimension.');
    end

    if nargin < 3
        dim = find(size(g)==length(g),true,'first');
    end

    if ~any(dim == [1,2,3])
        error('Convolution dimension must be 1, 2, or 3.');
    end

    if ~mod(length(g),2)
        g = cat(dim,g,0);
    end

    if ~any(ndims(x) == [2,3])
        error('x must have dimension 2 or 3.');
    end
    
    if size(x,dim) < length(g)-1
        error('size(x,dim) must be at least length(g)-1.');
    end
    
    %----------------------------------------------------------------------

    permx = (dim~=1);
    if permx
        p = 1:ndims(x);
        p(1) = dim;
        p(dim) = 1;
        x = permute(x,p);
        dim = 1;
    end

    if isreal(x)
        y = conv_even_per_d(x,g(:),dim);
    else
        y = conv_even_per_cd(x,g(:),dim);
    end
    
    if permx
        y = ipermute(y,p);
    end

end
