function Apar = projection(a, b, idA, idB)
%PROJECTION  Vector component of A parallel to B.
%   This function exploits the MULTIPROD engine (MATLAB Central, file
%   #8773), which enables multiple products and array expansion (AX).
%
%   When A and B are vectors (e.g. P×1, 1×P, or 1×1×P arrays):
%
%       Apar = PROJECTION(A, B) returns the projection of A on the axis
%       along which B lies. A and B must have the same length.
%
%   More generally, when A and B are arrays of any size containing one or
%   more vectors:
%
%       Apar = PROJECTION(A, B) is equivalent to 
%       Apar = PROJECTION(A, B, IDA, IDB), where IDA and IDB are the first
%       non-singleton dimensions of A and B, respectively.
%
%       Apar = PROJECTION(A, B, DIM) is equivalent to 
%       Apar = PROJECTION(A, B, IDA, IDB), where IDA = IDB = DIM.
%
%       Apar = PROJECTION(A, B, IDA, IDB) operates on the vectors contained
%       along dimension IDA of A and dimension IDB of B, and returns the
%       projections of the vectors in A on the axes along which the vectors
%       in B lie. These vectors must have the same length P = SIZE(A,IDA) =
%       SIZE(B,IDB). A and B are viewed as "block arrays". IDA and IDB are
%       referred to as their "internal dimensions" (IDs). For instance, a
%       5×6×2 array may be viewed as an array containing twelve 5-element
%       blocks. In this case, its size is denoted by (5)×6×2, and its ID is
%       1. Since AX is enabled, A and B may have different size, and IDA
%       may not coincide with IDB (see MULTIPROD).
%
%       Input and output format:
%           Array     Block size     Internal dimension
%           ---------------------------------------------------------------
%           A         P  (1-D)       IDA
%           B         P  (1-D)       IDB
%           Apar      P  (1-D)       MAX(IDA, IDB)
%           ---------------------------------------------------------------
%           If SIZE(A)==SIZE(B) and IDA==IDB, then SIZE(C)=SIZE(A)=SIZE(B).
%
%   Examples:
%      If A and B are both ........ (5)×6×2 array of vectors,
%      Apar = PROJECTION(A, B) is a (5)×6×2 array of vectors. 
%
%      Thirty vectors contained in A are projected on the axis along which
%      a single vector B lies: 
%      If  A is ....................... a 5×6×(3) array of 30 vectors,
%      and B is ....................... a (3)×1   vector
%      Apar = PROJECTION(A, B, 3, 1) is a 5×6×(3) array of 30 vectors. 
%
%   See also REJECTION, MAGN, UNIT, DOT2, CROSS2, CROSSDIV, OUTER.

% $ Version: 2.0 $
% CODE      by:                 Paolo de Leva (IUSM, Rome, IT) 2009 Jan 25
% COMMENTS  by:                 Code author                    2009 Feb 13
% OUTPUT    tested by:          Code author                    2009 Feb 13
% -------------------------------------------------------------------------

% Allow 2 to 4 input arguments
error( nargchk(2, 4, nargin) ); 

% Setting IDA and/or IDB
switch nargin
    case 2
        idA0 = find(size(a)>1, 1, 'first'); % First non-singleton dim.
        idB0 = find(size(b)>1, 1, 'first'); % ([] if the array is a scalar)
        idA = max([idA0, 1]); % IDA = 1 if A is a scalar
        idB = max([idB0, 1]);
    case 3
        idB = idA;
end

unitB = unit(b, idB); % Versor of B
scalarApar = dot2(a, unitB, idA, idB); % Scalar component
Apar = multiprod(scalarApar, unitB, max(idA, idB), idB); % Vector component