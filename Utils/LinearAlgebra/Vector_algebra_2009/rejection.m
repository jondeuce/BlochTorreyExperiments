function Aorth = rejection(a, b, idA, idB)
%REJECTION  Vector component of A orthogonal to B.
%   This function exploits the MULTIPROD engine (MATLAB Central, file
%   #8773), which enables multiple products and array expansion (AX).
%
%   When A and B are vectors (e.g. P×1, 1×P, or 1×1×P arrays):
%
%       Aorth = REJECTION(A, B) returns the rejection of A from the axis
%       along which B lies. The rejection of a vector from an axis is its
%       projection on a plane orthogonal to that axis. A and B must have
%       the same length.
%
%   More generally, when A and B are arrays of any size containing one or
%   more vectors:
%
%       Aorth = REJECTION(A, B) is equivalent to 
%       Aorth = REJECTION(A, B, IDA, IDB), where IDA and IDB are the first
%       non-singleton dimensions of A and B, respectively.
%
%       Aorth = REJECTION(A, B, DIM) is equivalent to 
%       Aorth = REJECTION(A, B, IDA, IDB), where IDA = IDB = DIM.
%
%       Aorth = REJECTION(A, B, IDA, IDB) operates on the vectors contained
%       along dimension IDA of A and dimension IDB of B, and returns the
%       rejections of the vectors in A from the axes along which the
%       vectors in B lie. These vectors must have the same length 
%       P = SIZE(A,IDA) = SIZE(B,IDB). A and B are viewed as "block
%       arrays". IDA and IDB are referred to as their "internal dimensions"
%       (IDs). For instance, a 5×6×2 array may be viewed as an array
%       containing twelve 5-element blocks. In this case, its size is
%       denoted by (5)×6×2, and its ID is 1. Since AX is enabled, A and B
%       may have different size, and IDA may not coincide with IDB (see
%       MULTIPROD).
%
%       Input and output format:
%           Array     Block size     Internal dimension
%           ---------------------------------------------------------------
%           A         P  (1-D)       IDA
%           B         P  (1-D)       IDB
%           Aorth     P  (1-D)       MAX(IDA, IDB)
%           ---------------------------------------------------------------
%           If SIZE(A)==SIZE(B) and IDA==IDB, then SIZE(C)=SIZE(A)=SIZE(B).
%
%   Examples:
%      If A and B are both ....... (5)×6×2 array of vectors,
%      Apar = REJECTION(A, B) is a (5)×6×2 array of vectors. 
%
%      Thirty vectors contained in A are rejected from the axis along which
%      a single vector B lies: 
%      If  A is ....................... a 5×6×(3) array of 30 vectors,
%      and B is ....................... a (3)×1   vector
%      Aorth = REJECTION(A, B, 3, 1) is a 5×6×(3) array of 30 vectors. 
%
%   See also PROJECTION, MAGN, UNIT, DOT2, CROSS2, CROSSDIV, OUTER.

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

% Projection
Apar = projection(a, b, idA, idB);

% Rejection (a shift is needed only if idB > idA)
shiftA = max(0, idB-idA); 
Aorth = baxfun(@minus, a, Apar, shiftA); % (A - Apar)