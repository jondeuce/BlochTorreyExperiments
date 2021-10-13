function [ y ] = reshapeVec( x, dim )
%RESHAPEVEC Reshapes the array 'x' to be a vector of length numel(x) along
%dimension dim

switch dim
    case 1
        y	=   x(:);
    case 2
        y	=   x(:).';
    case 3
        y	=   reshape( x(:), 1, 1, [] );
    otherwise
        ysiz        =   ones(1,dim);
        ysiz(dim)	=   numel(x);
        y           =   reshape( x(:), ysiz );
end

end

