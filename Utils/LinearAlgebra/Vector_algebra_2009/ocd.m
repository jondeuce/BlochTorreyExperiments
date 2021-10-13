function Aorth = ocd(c, b, idC, idB)
%OCD  Orthogonal cross division.
%   This function exploits the BAXFUN engine (MATLAB Central, file #23084),
%   which enables multiple cross divisions and array expansion (AX).
%
%   When B and C are 3-element vectors (e.g. 3×1, 1×3, or 1×1×3 arrays):
%
%       Aorth = OCD(C, B) returns the orthogonal cross division C /o B.
%
%   More generally, when B and C are arrays of any size containing one or
%   more 3-element vectors:
%
%       Aorth = OCD(C, B) is equivalent to Aorth = OCD(C, B, IDC, IDB),
%       where IDA and IDB are the first dimensions of A and B whose length
%       is 3. 
%
%       Aorth = OCD(C, B, DIM) is equivalent to 
%       Aorth = OCD(C, B, IDC, IDB), where IDC = IDB = DIM.  
%   
%       Aorth = OCD(C, B, IDC, IDB) returns the orthogonal cross divisions
%       between the vectors contained in C along dimension IDC and those
%       contained in B along dimension IDB. These vectors must have 3
%       elements. C and B are viewed as "block arrays". IDC and IDB are
%       referred to as their "internal dimensions" (IDs). For instance, a
%       3×6×2 array may be viewed as an array containing twelve 3-element
%       blocks. In this case, its size is denoted by (3)×6×2, and its ID is
%       1. Since AX is enabled, C and B may have different size, and IDC
%       may not coincide with IDB (see MULTIPROD).
%
%       Input and output format:
%           Array     Block size     Internal dimension
%           ---------------------------------------------------------------
%           C         3  (1-D)       IDC
%           B         3  (1-D)       IDB
%           A         3  (1-D)       MAX(IDC, IDB)
%           ---------------------------------------------------------------
%           If SIZE(B)==SIZE(C) and IDB==IDC, then SIZE(A)=SIZE(B)=SIZE(C).
%
%   Two main kinds of cross divisions exist: indefinite (ICD) and definite
%   (DCD). See the help text of function CROSSDIV for details. The
%   orthogonal cross division (OCD) is the simplest form of DCD. It is
%   equal to the known and unique orthogonal term of an ICD:
%                      C /o B = Aorth = B x C / (B * B)                 (1)
%   When Aorth coincides with A (see example),
%                                C /o B = A                             (2)
%
%   Minimum-input OCDs
%       One of the components of B or C can be omitted, provided it is
%       relative to an axis non-orthogonal to C or B. Since B and C are by
%       definition orthogonal, the missing component can be computed by
%       solving the equation B * C = 0. To perform a minimum-input OCD, use
%       Aorth = CROSSDIV(TYPE, C, B, pi/2, IDC, IDB), where TYPE has one of
%       the following values (see CROSSDIV for details):
%           '+t-Bx'   (Unknown component of vector B)
%           '+t-By'
%           '+t-Bz'
%           '+t-Cx'   (Unknown component of vector C)
%           '+t-Cy'
%           '+t-Cz'
%
%   Warning:
%       By definition, in an OCD (C /o B) the divisor (B) must be the
%       second operand of a cross product (A x B = C). When you use as
%       divisor the first operand (A) of the cross product, rather than the
%       second, the OCD does not yield the expected result (Borth). Since 
%       A x B = - B x A, it yields the opposite result:
%                              C /o A = - Borth                         (3)
%
%   Reference:
%       de Leva, P (2008). Anticrossproducts and cross divisions. 
%       Journal of Biomechanics, 8, 1790-1800.
%
%   Examples:
%       V is the derivative of a vector R which rotates with an angular
%       velocity OMEGA. V has two components, parallel and orthogonal to R
%       (V = Vpar + Vorth). Since Vorth = OMEGA x R (Poisson's formula),
%       then
%                           OMEGA = OCD(Vorth, R)      [OMEGA = Vorth /o R]
%       Since R x Vorth = R x V, this is equivalent to
%                           OMEGA = OCD(V, R)              [OMEGA = V /o R]
%       valid in 3D even when the magnitude of R is not constant (Vpar~=0). 
%
%       If C and B are both (3)×6×2 arrays of vectors,
%       C = OCD(C, B) is  a (3)×6×2 array  of vectors.
%
%       A single vector C is divided by thirty vectors contained in B: 
%       If  C is ............. a (3)×1   vector,
%       and B is ............. a 5×6×(3) array of 30 vectors,
%       C = OCD(C, B, 1, 3) is a 5×6×(3) array of 30 vectors.
%
%   See also CROSSDIV, DOT2, CROSS2, OUTER, MAGN, UNIT, 
%            PROJECTION, REJECTION, TESTOCD.

% $ Version: 3.0 $
% CODE      by:                 Paolo de Leva (IUSM, Rome, IT) 2009 Feb 2
% COMMENTS  by:                 Code author                    2009 Feb 26
% OUTPUT    tested by:          Code author                    2009 Feb 26
% -------------------------------------------------------------------------

% Allow 2 to 4 input arguments
error( nargchk(2, 4, nargin) ); 

% Setting IDC and/or IDB
switch nargin
    case 2        
        idC = find(size(c)==3, 1, 'first'); % First dim. of length 3
        idB = find(size(b)==3, 1, 'first');
        if isempty(idC) || isempty(idB)
            error('OCD:InvalidSize',...
                  'C and B must have at least one dimension of length 3.');
        end        
    case 3
        idB = idC;
end

% 1 - Cross product (B x C)
%     (This will issue an error if SIZE(B,IDB)~=3 or SIZE(C,IDC)~=3)
BxC = cross2(b, c, idB, idC);

% 2 - Dot product (B * B)
BB = dot(b, b, idB);

% 3 - Orthogonal term of cross division C // B.
%     (BAXFUN replicates B2 3 times along its singleton dimension IDB)
shiftB = max(0, idC-idB);
Aorth = baxfun(@rdivide, BxC, BB, 0, shiftB);

% NOTE: For vectors with null magnitude, the latter divison (by zero) will
% cause MATLAB to issue a warning. The respective normalized vector will be
% composed of NaNs.