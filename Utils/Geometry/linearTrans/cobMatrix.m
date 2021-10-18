function T = cobMatrix( P, Q, flag )
%COBMATRIX outputs change of basis matrix T
% 
% Inputs:
%   -P: matrix of current basis vectors
%   -Q: matrix of desired basis vectors
%   -flag: flag to indicate whether basis represented as row/column vectors:
%       -if flag==true or flag=='cols', basis interpreted as columns
%       -if flag==false or flag=='rows', basis interpreted as rows
% 
% Outputs:
%   -T: change of basis matrix
%       -if basis given as columns, then Q = T * P
%       -if basis given as rows, then Q = P * T
% 
% Notes:
%   -will work for square invertible matrices P of any dimension
%   -T is not necessarily a rotation matrix

if isa(flag,'logical')
    if flag
        T=Q/P;
    else
        T=P\Q;
    end
elseif isa(flag,'char')
    switch upper(flag)
        case 'COLS'
            T=Q/P;            
        case 'ROWS'
            T=P\Q;
        otherwise
            error('ERROR (changeOfBasis): valid flags: ''rows'' or ''cols''');
    end
else
    error('ERROR (changeOfBasis): flag must be a logical or a string');
end

end

