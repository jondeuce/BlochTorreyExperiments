function [ y ] = conv_per( x, g, dim )
%CONV_PER Convolves the 2D or 3D complex array x with the kernel g. g must
% be a vector (have only one non-singleton dimension), and have odd length.
% If g has even length it will be padded to odd length with a zero.

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

    permx = (dim==3);
    %permx = (dim~=1);
    if permx
        p = 1:ndims(x);
        p(1) = dim;
        p(dim) = 1;
        x = permute(x,p);
        dim = 1;
    end

    if isreal(x)
        y = conv_per_d(x,g(:),dim);
    else
        y = conv_per_cd(x,g(:),dim);
    end

    if permx
        y = ipermute(y,p);
    end

end
