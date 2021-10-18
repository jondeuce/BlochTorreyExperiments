function B = uaxfun( f, A, xdim, ysiz )
%UAXFUN Unary array expansion function. 
% 
% INPUT ARGUMENTS
%   f:     -function to apply to 'A' along the dimension(s) xdim
%   A:     -ND array which the function 'f' will be applied to
%   xdim:  -dimension(s) on which to apply the function 'f' to
%   ysiz:  -(optional) dimension(s) of the output of the function 'f';
%           if ysiz is not provided, f will be evaluated once to determine
%           the dimensions of the output
%              -if ndims(output) = ndims(input)
%                  => ndims(B) = ndims(A)
%              -if ndims(output) < ndims(input)
%                  => ndims(B) < ndims(A), with corresponding missing
%                     dimensions of A squeezed out
%              -if ndims(output) > ndims(input)
%                  => ndims(B) > ndims(A), with extra dimensions added to
%                     appended to the end of B
% 
%   NOTE:  -output sizes may vary in all cases, depending on 'f'
%          -length of 'xdim' must be either 1 or 2 only, that is 'f' must
%           take either a 1D or 2D array as an input, and return an ND-
%           array as an output
% 
% OUTPUT ARGUMENTS
%   B:     -ND array given by the application of 'f' on 'A' along 'xdim'
%              -dimensions of 'B' will be the same as 'A', possibly with
%               the exception of those dimensions given by 'xdim' and any
%               additional dimensions added if ndims(output) > ndims(input)

% size and dimension of input array
siz = size(A);
dim = ndims(A);

% check for valid xdim
if ~any( length(xdim) == [1,2] )
    error('ERROR: ysiz must have length 1 or 2 only');
end

% check ysiz
if length(ysiz) == 1
    % default to column vector
    ysiz = [ysiz, 1];
end

% dimensions of input/output
m = length( xdim );
n = get_ydim( ysiz );

if ~( n > 0 )
    error('ERROR: ysiz must be non-empty');
end

outsize = get_outsize( siz, xdim, ysiz );
resiz	= get_resize( siz, xdim, ysiz );

f = @(x) reshape( f(squeeze(x)), resiz );
B = cell2mat( ...
        cellfun( f, mat2cell( A, outsize{:} ), 'uniformoutput', false ) ...
    );

end

function resiz = get_resize( siz, xdim, ysiz )

dim = length(siz);
m = length(xdim);
n = get_ydim( ysiz );

switch m
    case 1
        xinds = ( (1:dim) == xdim(1) );
    case 2
        xinds = ( (1:dim) == xdim(1) ) | ( (1:dim) == xdim(2) );
    otherwise
        error('ERROR: xdim must have length 1 or 2 only.');
end

if all( ysiz == [1,1] )
    yinds = [true, false];
else
    yinds = ( ysiz ~= 1 );
end
resiz = ones(1,dim);

switch (n-m)
    case -1 % must be that n = 1, m = 2
        % contract along singleton dimension
        resiz( xinds( yinds) ) = ysiz( yinds );
        resiz( xinds(~yinds) ) = [];
    case 0 % either n = 1, m = 1 or n = 2, m = 2
        resiz( xinds ) = ysiz( yinds );
    otherwise % either n > m = 1, or n > m = 2;
        ysqueeze = ysiz( yinds );
        resiz( xinds ) = ysqueeze(1:m);
        resiz = [ resiz, ysqueeze(m+1:end) ];
end

end

function outsize = get_outsize( siz, xdim, ysiz )

dim = length(siz);
m = length(xdim);
% n = get_ydim( ysiz );

outsize = num2cell(siz);

for ii = 1:dim
    if ( m == 1 && ii ~= xdim ) || ...
            ( m == 2 && ~( ii == xdim(1) || ii == xdim(2) ) )
        % all dimensions except the input dimensions should remain the same
        outsize{ii} = ones( siz(ii), 1 );
    end
end

end

function n = get_ydim( ysiz )

if all( ysiz == [1,1] )
    % output is a scalar
    n = 1;
else
    % dim is the number of non-singleton dimensions
    n = sum( ysiz ~= 1 );
end

end

