function V = nullVectors( E1 )
%NULLVECTORS Gets the set of orthonormal null vectors associated with 'e1'
% 
% INPUT ARGUMENTS
%   E1:    -matrix of vectors for which the set of orthonormal null vectors
%           will be found.
%          -size must be either M x N, M x 1 x N, or M x 1 x N x P x ...,
%           where M is the length of an individual vector
%               -i.e. vectors of interest must be 'column vectors'
% 
% OUTPUT ARGUMENTS
%   V:     -matrix of orthonormal null vectors
%          -size will be one of M x N x (M-1), M x (M-1) x N, or
%           M x (M-1) x N x P x ..., depending on the size of E1

siz = size(E1);
dim = ndims(E1);

if dim == 2
    % Handle 2-dimensional case separately
    M = siz(1);
    N = siz(2);
    
    if N == 1
        V = fnull(E1);
        return
    end
    
    outsize = { M, ones(N,1) };
    func = @(e1) reshape( null(e1'), [M,1,M-1] );
    V = cell2mat(	cellfun( func, mat2cell( E1, outsize{:} ),	...
        'uniformoutput', false	)                       ...
        );
    return
    
else
    if siz(2) ~= 1
    	error('ERROR: invalid input dimension');
    end
end

outsize     = num2cell( size(E1) );
outsize{3}	= ones( size(E1,3), 1 );

if dim >= 4
    for n = 4:dim
        outsize{n} = ones( size(E1,n), 1 );
    end
end

V = cell2mat(	cellfun( @fnull, mat2cell( E1, outsize{:} ),	...
    'uniformoutput', false	)                       ...
    );

end

function v = fnull(e1)

v = null(e1');

end