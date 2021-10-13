function X = cobTransform( X, T, flag )
%COBTRANSFORM applies the transformation matrix T to the matrix X
% 
% Inputs:
%   -X: n x 3 or 3 x n point cloud (specified by flag)
%   -T: change of basis transformation matrix
%   -flag: flag to indicate whether basis represented as row/column vectors:
%       -if flag==true or flag=='cols', basis interpreted as columns
%       -if flag==false or flag=='rows', basis interpreted as rows
% 
% Outputs:
%   -X: input point cloud X under new basis
%       -if basis given as columns, then X = T * X
%       -if basis given as rows, then X = X * T
% 
% See also: COBMATRIX

if isa(flag,'logical')
    if flag
        X=T*X;
    else
        X=X*T;
    end
elseif isa(flag,'char')
    switch upper(flag)
        case 'COLS'
            X=T*X;           
        case 'ROWS'
            X=X*T;
        otherwise
            error('ERROR (changeOfBasis): valid flags: ''rows'' or ''cols''');
    end
else
    error('ERROR (changeOfBasis): flag must be a logical or a string');
end
end

