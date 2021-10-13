function c = outer(a, b, idA, idB)
%OUTER  Vector outer product.
%   This function exploits the MULTIPROD engine (MATLAB Central, file
%   #8773), which enables multiple products and array expansion (AX).
%
%   When A and B are vectors (e.g. P�1, 1�P, or 1�1�P arrays):
%
%       C = OUTER(A, B) returns their outer product. A and B may have
%       different lengths P and Q. If A and B are both column vectors, C is
%       a P�Q matrix, and OUTER(A, B) is the same as CONJ(A) * B'.
%       Otherwise, C is a multi-dimensional array containing a single PxQ
%       matrix (e.g. a 1�P�Q array, if A and B are row vectors; see below).
%       C = OUTER(A(:), B(:)) guarantees that C is a matrix.
%
%   More generally, when A and B are arrays of any size containing one or
%   more vectors:
%
%       C = OUTER(A, B) is equivalent to C = OUTER(A, B, IDA, IDB), where
%       IDA and IDB are the first non-singleton dimensions of A and B,
%       respectively.
%
%       C = OUTER(A, B, DIM) is equivalent to C = OUTER(A, B, IDA, IDB),
%       where IDA = IDB = DIM.
%
%       C = OUTER(A, B, IDA, IDB) returns the outer products between the
%       vectors contained in A along dimension IDA and those contained in B
%       along dimension IDB. These vectors may have different lengths 
%       P = SIZE(A,IDA) and Q = SIZE(B,IDB). A and B are viewed as "block
%       arrays". IDA and IDB are referred to as their "internal dimensions"
%       (IDs). For instance, a 5�6�2 array may be viewed as an array
%       containing twelve 5-element blocks. In this case, its size is
%       denoted by (5)�6�2, and its ID is 1. Since AX is enabled, the
%       "external dimensions" of A and B may have different size, and IDA
%       may not coincide with IDB (see MULTIPROD).
%
%       C = OUTER(A, B, IDA, IDB) just calls the instruction
%       C = MULTIPROD(CONJ(A), B, [IDA 0], [0 IDB]), which turns the
%       vectors found in CONJ(A) and B into P�1 and 1�Q matrices,
%       respectively, then multiplies them obtaining P�Q matrices.
%
%       Input and output format (see MULTIPROD, syntax 4c):
%           Array     Block size     Internal dimension(s)
%           ------------------------------------------------------
%           A         P    (1-D)     IDA
%           B         Q    (1-D)     IDB
%           C         P�Q  (2-D)     MAX([IDA IDA+1], [IDB IDB+1])
%           ------------------------------------------------------
%
%   Examples:
%       If  A is ......... a   (5)�6�2 array of vectors,
%       and B is ......... a   (3)�6�2 array of vectors,
%       C = OUTER(A, B) is a (5�3)�6�2 array of matrices.
%
%       A single vector B multiplies thirty vectors contained in A: 
%       If  A is ............... a 5�6�(2)   array of 30 vectors,
%       and B is ............... a (3)�1     vector,
%       C = OUTER(A, B, 3, 1) is a 5�6�(2�3) array of 30 matrices.
%
%   See also DOT2, CROSS2, CROSSDIV, MAGN, UNIT, PROJECTION, REJECTION,
%            MULTIPROD, TESTOUTER.

% $ Version: 2.0 $
% CODE      by:                 Paolo de Leva (IUSM, Rome, IT) 2009 Feb 1
% COMMENTS  by:                 Code author                    2009 Feb 12
% OUTPUT    tested by:          Code author                    2009 Feb 12
% -------------------------------------------------------------------------

% Allow 2 to 4 input arguments
narginchk(2, 4) ;

% Setting IDA and/or IDB
switch nargin
    case 2
        idA0 = find(size(a)>1, 1, 'first'); % First non-singleton dim.
        idB0 = find(size(b)>1, 1, 'first'); % ([] if the array is a scalar)
        idA = max([idA0, 1]); % IDA = 1 if A is empty or a scalar
        idB = max([idB0, 1]);
    case 3
        idB = idA;
end

c = multiprod(conj(a), b, [idA 0], [0 idB]);