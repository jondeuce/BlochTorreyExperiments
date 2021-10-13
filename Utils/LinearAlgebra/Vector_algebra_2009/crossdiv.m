function a = crossdiv(varargin)
%CROSSDIV  Vector cross division.
%   This function exploits the BAXFUN engine (MATLAB Central, file #23084),
%   which enables multiple cross divisions and array expansion (AX).
%
%   Orthogonal cross division (OCD):
%       Aorth = CROSSDIV(C, B) is equivalent to Aorth = OCD(C, B)
%       Aorth = CROSSDIV(C, B, ...) is equivalent to Aorth = OCD(C, B, ...)
%
%   When B and C are 3-element vectors (e.g. 3�1, 1�3, or 1�1�3 arrays):
%
%       A = CROSSDIV(TYPE, C, B, EXTRA), where C = A x B, returns a
%       definite cross division C / B of type TYPE, with additional input
%       argument EXTRA. 
%
%   More generally, when B and C are arrays of any size containing one or
%   more 3-element vectors:
%
%       A = CROSSDIV(TYPE, C, B, EXTRA) is equivalent to 
%       A = CROSSDIV(TYPE, C, B, EXTRA, IDC, IDB), where
%       IDA and IDB are the first dimensions of A and B whose length is 3.
%
%       A = CROSSDIV(TYPE, C, B, EXTRA, DIM) is equivalent to 
%       A = CROSSDIV(TYPE, C, B, EXTRA, IDC, IDB), where IDC = IDB = DIM. 
%
%       A = CROSSDIV(TYPE, C, B, EXTRA, IDC, IDB), where C = A x B, returns
%       the definite cross divisions between the vectors contained in C
%       along dimension IDC and those contained in B along dimension IDB.
%       These vectors must have 3 elements. C and B are viewed as "block
%       arrays". IDC and IDB are referred to as their "internal dimensions"
%       (IDs). For instance, a 3�6�2 array may be viewed as an array
%       containing twelve 3-el. blocks. In this case, its size is denoted
%       by (3)�6�2, and its ID is 1. Since AX is enabled, C and B may have
%       different size, and IDC may not coincide with IDB (see MULTIPROD).
%
%       TYPE      A string containing one specifier for the type of
%                 additional input (EXTRA), and optionally one specifier
%                 for the type of missing input, for minimum-input CDs.
%                 White spaces are ignored. TYPE is case insensitive. 
%                 Specifiers for the type of EXTRA:
%                     '+s'                 Cross division coefficient.
%                     '+theta' (or '+t')   Angle between A and B (radians).
%                     '+Ax'    (or '+x')   Known component of vector A.
%                     '+Ay'    (or '+y')
%                     '+Az'    (or '+z')
%                 Specifiers for the type of missing input:
%                     ''                   No missing input.
%                     '-Bx'                Unknown component of vector B.
%                     '-By'
%                     '-Bz'
%                     '-Cx' (or '-x')      Unknown component of vector C.
%                     '-Cy' (or '-y')
%                     '-Cz' (or '-z')
%       A, B, C   Array     Block size     Internal dimension
%                 ---------------------------------------------------------
%                 C         3  (1-D)       IDC
%                 B         3  (1-D)       IDB
%                 A         3  (1-D)       MAX(IDC, IDB)
%                 ---------------------------------------------------------
%                 For minimum-input CDs, an arbitrary value (e.g. NaN) must
%                 be assigned to the missing component of each vector
%                 contained in B or C. If SIZE(B) == SIZE(C) and 
%                 IDB == IDC, then SIZE(A) = SIZE(B) = SIZE(C).
%       EXTRA     Additional input value(s) (see TYPE). EXTRA must be 
%                 either a scalar or an array of the same length as A along
%                 all dimensions except IDA. SIZE(EXTRA, IDA) must be 1.
%       IDC, IDB  The "internal dimensions" along which the vectors are
%                 contained in A and B, respectively.
%
%   Two main kinds of cross divisions (CDs) exist: indefinite (ICDs) and 
%   definite (DCDs). DCDs allow to write in a short vectorial form
%   equations such as those used to compute the point of application of a
%   force based on dynamometric data (see examples). The definition of the
%   DCD is based on that of the ICD. The symbol of the ICD is //. As well
%   as indefinite integrals yield infinite antiderivatives, ICDs yield
%   infinite anticrossproducts (antiCPs). There are infinite vectors X
%   meeting the condition X x B = C, all laying on a plane normal to C. If
%   they include A, they are called the "antiCPs of A x B�, where A x B =
%   C. These antiCPs are determined by C // B, defined as follows:
%                          C // B = X = Aorth + Xpar                    (1)
%                           Aorth = B x C / (B * B)       (Orthogonal term)
%                            Xpar = S * UNIT(B)             (Parallel term)
%   Each antiCP X shares with A its component Aorth = Xorth, orthogonal to
%   both B and C, and differs from the others only by Xpar, its component
%   parallel to B. Since Aorth is unique, it can be determined as shown
%   above. Notice that Aorth is one of the antiCPs (Aorth x B = C).
%   Although Xpar is nonunique, it can be partially determined. Being it
%   parallel to B, it is equal to S * UNIT(B), where S is a scaling factor,
%   called "cross division coefficient". Thus, the only fully undetermined
%   value in equation 1 is S. Luckily, in some cases S is known or it can
%   be determined otherwise, and this allows for the definition of four
%   kinds of DCDs (see below). In the definition of an ICD, S has the same
%   role as the arbitrary integration constant in an indefinite integral.
%
%   Orthogonal cross division (OCD)
%       The OCD is the simplest form of DCD. It is determined by setting
%       S = 0, and is equal to the unique orthogonal term of an ICD:
%                      C /o B = Aorth = B x C / (B * B)                 (2)
%       When the orthogonal antiCP Aorth happens to coincide with A,
%                                C /o B = A                            (2b)
%
%   Basic DCD
%       When the value Sa for S which uniquely identifies A in eq. 1 is
%       known, A can be fully determined as follows:
%               C /s B = Aorth + Apar = C /o B + Sa * UNIT(B) = A       (3)
%
%   Computed-coefficient DCDs
%       In equation 3, Sa can be computed as a function of:
%           - THETA (angle between A and B) or
%           - a component of A along an axis non-orthogonal to B.
%
%   Minimum-input DCDs
%       In any of the above mentioned DCDs, one of the components of B or C
%       can be omitted, provided it is relative to an axis non-orthogonal 
%       to C or B. Since B and C are by definition orthogonal, the missing
%       component can be computed by solving the equation B * C = 0.
%
%   Warning:
%       By definition, in an ICD (C // B) the divisor (B) must be the 2nd
%       operand of a cross product (A x B = C). When you use as divisor the
%       1st operand (A) of the cross product, rather than the 2nd, the ICD
%       does not yield the expected set of antiCPs (X including B). 
%       Since A x B = -B x A, it yields the opposite set (�X including �B):
%                     C // A = - X = - Borth - S * UNIT(A)              (4)
%       Being derived by the ICD, all the DCDs share the same limitation.
%       E.g.,
%                             C /o A = - Borth                          (5)
%
%   Reference:
%       de Leva, P (2008). Anticrossproducts and cross divisions. 
%       Journal of Biomechanics, 8, 1790-1800.
%
%   Examples:
%      If the moment of a force F is R x F = M:
%          1) R = CROSSDIV('+t',   M, F, THETA)   [Computed-coeff. DCD] 
%          2) R = CROSSDIV('+z',   M, F, Rz)      [Computed-coeff. DCD] 
%          3) R = CROSSDIV('+z-z', M, F, Rz)      [Minimum-input DCD] 
%          4) F = - CROSSDIV('+t', M, R, THETA)   [Equation 4] 
%       See also the examples in the help text of function OCD.
%       NOTE 1 - When Rz is used as input argument, axis z must be
%                non-orthogonal to F (i.e. Fz must be non-null).
%       NOTE 2 - The third instruction can be used to compute the point of
%                application of a force based on the output of a force
%                plate (typically, z is vertical, Rz known, Mz unknown).
%
%       If C and B are both ................ (3)�6�2 arrays of vectors,
%       and EXTRA is either a scalar or .. a (1)�6�2 array
%       C = CROSSDIV(TYPE, C, B, EXTRA) is a (3)�6�2 array  of vectors.
%
%       A single vector C is divided by thirty vectors contained in B: 
%       If  C is .......................... a (3)�1   vector,
%           B is .......................... a 5�6�(3) array of 30 vectors,
%       and EXTRA is either a scalar or ... a 5�6�(1) array
%       C = CROSSDIV(TYPE,C,B,EXTRA,1,3) is a 5�6�(3) array of 30 vectors.
%
%   See also OCD, DOT2, CROSS2, OUTER, MAGN, UNIT, PROJECTION, REJECTION,
%            TESTXDIV.

% $ Version: 3.0 $
% CODE      by:                 Paolo de Leva (IUSM, Rome, IT) 2006 Jul 20
%           optimized by:       Code author                    2009 Feb 12
% COMMENTS  by:                 Code author                    2009 Feb 26
% OUTPUT    tested by:          Code author                    2009 Feb 26
% -------------------------------------------------------------------------

% SIMPLEST CASE: A simple OCD (no missing value)
    if ~ischar(varargin{1})
        a = ocd(varargin{:}); 
        return
    end

% Checking and adjusting input arguments
    narginchk(4, 6) ;
    [type, c, b, extra]  = varargin{1:4};
    type = type( type~=' ' ); % Removing white spaces
    type = lower(type);       % Lower case
    sizeC = size(c);
    sizeB = size(b);
    sizeE = size(extra);
    
    % Dividing TYPE into two substrings
    findplus  = strfind(type, '+'); % Mandatory specifier
    findminus = strfind(type, '-'); % Optional  specifier
    if length(findplus)~=1 || length(findminus)>1
        error('CROSSDIV:InvalidType', ['Invalid TYPE: ', type]);
    elseif isempty( findminus )
        extratype = type;
        missingtype = '';
    elseif findplus > 1
        missingtype = type( 1 : findplus-1 );
        extratype   = type( findplus : end );
    else % findminus > 1
        extratype   = type( 1 : findminus-1 );
        missingtype = type( findminus : end );
    end

    % Setting flags specifying type of EXTRA
    switch extratype
        case {'+ax',    '+x'}, extraflag = 'x'; extraaxis = 1;
        case {'+ay',    '+y'}, extraflag = 'y'; extraaxis = 2;
        case {'+az',    '+z'}, extraflag = 'z'; extraaxis = 3;
        case            '+s' , extraflag = 's'; extraaxis = 0;
        case {'+theta', '+t'}, extraflag = 't'; extraaxis = 0;
        otherwise, error('CROSSDIV:InvalidType', ['Invalid TYPE: ', type]);
    end

    % Setting flag specifying type of missing value
    switch missingtype
        case '',            missingaxis = 0;
        case '-bx',         missingaxis = 1;
        case '-by',         missingaxis = 2;
        case '-bz',         missingaxis = 3;
        case {'-cx', '-x'}, missingaxis = 4;
        case {'-cy', '-y'}, missingaxis = 5;
        case {'-cz', '-z'}, missingaxis = 6;
        otherwise, error('CROSSDIV:InvalidType', ['Invalid TYPE: ', type]);
    end

    % Setting IDC and/or IDB
    switch nargin
        case 4        
            idC = find(size(c)==3, 1, 'first'); % First dim. of length 3
            idB = find(size(b)==3, 1, 'first');
            if isempty(idC) || isempty(idB)
                error('CROSSDIV:InvalidSize',...
                      'C and B must have at least one dimension of length 3.');
            end        
        case 5
            idC = varargin{5};
            idB = idC;
        case 6
            [idC, idB] = varargin{[5 6]};
    end

    % Checking block size
    if (sizeC(idC)~=3) || (sizeB(idB)~=3),
        error('CROSSDIV:InvalidBlockSize',...
             ['C and B must be of length 3 in the dimensions\n'...
              'in which the cross division is taken.'])
    end

    % Dimension shift (first step of array expansion)
    %     NOTE: The BAXFUN engine is implemented here without calling
    %            BAXFUN, to avoid repeated shift of B and C
    diff = idC - idB;
    if diff == 0
        id = idB;  % ID = IDA = IDB = IDC
    elseif diff > 0
        id = idC;
        sizeB = [ones(1, diff) sizeB];
        b = reshape(b, sizeB);
        % if sizeB(end)==1, sizeB = sizeB(end-1); end % when B is 3�1
    else % diff < 0
        id = idB;
        sizeC = [ones(1,-diff) sizeC];
        c = reshape(c, sizeC);
        % if sizeC(end)==1, sizeC = sizeC(end-1); end % when C is 3�1
    end
    
    % Computing size of A
    NdimsB = length(sizeB);
    NdimsC = length(sizeC);
    NdimsA = max(NdimsB, NdimsC);
    NsinglB = NdimsA - NdimsB; % Number of added trailing singletons
    NsinglC = NdimsA - NdimsC;
    adjsizeB = [sizeB ones(1,NsinglB)];    
    adjsizeC = [sizeC ones(1,NsinglC)];    
    sizeA = max(adjsizeB, adjsizeC) .* (adjsizeB>0 & adjsizeC>0); 

    % Checking size of EXTRA
    fullsizeE = sizeA;
    fullsizeE(id) = 1;
    if ~isequal(sizeE, [1 1])
        NsinglE = NdimsA - length(sizeE);
        adjsizeE = [sizeE ones(1,NsinglE)];    
        if ~isequal(adjsizeE, fullsizeE)
            error('CROSSDIV:InvalidSizeExtra', 'Invalid size of EXTRA.');
        end
    end

% 1 - Minimum-input CDs
    if missingaxis
        notmissing = [1 2 3 1 2 3 1 2];
        notmissing1 = notmissing(missingaxis + 1);
        notmissing2 = notmissing(missingaxis + 2);
        
        % Vectorized indices
        % NOTE: If By or Cy is missing, it is important (in step 1a) that 
        %       the two NOTMISSING axes are [3 1] rather than [1 3]
        idxA = ivector(sizeA);
        idxB = ivector(sizeB); 
        idxC = ivector(sizeC); 

        % 1a - Simplified minimum-input CD
        if missingaxis > 3 && (missingaxis-3) == extraaxis
            a = zeros(sizeA);            
            % Known component of A
                idxA{id} = extraaxis;
                a(idxA{:}) = extra + zeros(fullsizeE);
            % A1 = EXTRA*B1 - C2
                idxA{id} = notmissing1;
                idxB{id} = notmissing1;
                idxC{id} = notmissing2;
                temp       = bsxfun( @times, extra, b(idxB{:}) );
                a(idxA{:}) = bsxfun( @minus, temp,  c(idxC{:}) );
            % A2 = EXTRA*B2 + C1)
                idxA{id} = notmissing2;
                idxB{id} = notmissing2;
                idxC{id} = notmissing1;
                temp       = bsxfun( @times, extra, b(idxB{:}) );
                a(idxA{:}) = bsxfun( @plus,  temp,  c(idxC{:}) );
            % Unknown components (A1,A2) = (A1,A2) / B0
                idxA{id} = [notmissing1 notmissing2];
                idxB{id} = extraaxis;
                a(idxA{:}) = bsxfun( @rdivide, a(idxA{:}), b(idxB{:}) );
            return

        % 1b - Computing the missing component of B or C by solving B*C = 0
        else
            idxB{id} = notmissing1;
            idxC{id} = notmissing1;
            temp = bsxfun( @times, b(idxB{:}), c(idxC{:}) ); % B1 * C1
            idxB{id} = notmissing2;
            idxC{id} = notmissing2;
            temp = temp + ...
                   bsxfun( @times, b(idxB{:}), c(idxC{:}) ); % B2 * C2

            % B0 = - (B1*C1 + B2*C2) / C0
            if missingaxis < 4
                if ~isequal(sizeA, sizeB)
                   b = bsxfun(@plus, b, zeros(sizeA)); % Expanding B
                   idxB = idxA;
                end
                idxB{id} = missingaxis; 
                idxC{id} = missingaxis; 
                b(idxB{:}) = - bsxfun( @rdivide, temp, c(idxC{:}) );

            % C0 = - (B1*C1 + B2*C2) / B0
            else % missingaxis > 3
                if ~isequal(sizeA, sizeC)
                   c = bsxfun(@plus, c, zeros(sizeA)); % Expanding C
                   idxC = idxA;
                end
                idxB{id} = missingaxis - 3; 
                idxC{id} = missingaxis - 3; 
                c(idxC{:}) = - bsxfun( @rdivide, temp, b(idxB{:}) );
            end
        end
    end
    clear temp

% 2 - Orthogonal cross division (A = Aorth = C /o B)
    a = ocd(c, b, id);

% 3 - Computed-coefficient DCD
    switch extraflag
        
        case 't' % Angle-input CD (Sa = MAGN(Aorth) / TAN(EXTRA))
            SimplifiedBasic = false;
            if  isequal(sizeE, [1 1]) && mod(extra, pi) == 0.5*pi
                return; % OCD (A = Aorth)
            end
            sa = bsxfun( @rdivide, magn(a, id), tan(extra) );

        case {'x','y','z'} % Component-input CD (SA0 = (EXTRA-Aorth0) / B0)            
            SimplifiedBasic = true;
            if ~missingaxis
                idxA = ivector(sizeA);
                idxB = ivector(sizeB);
            end
            idxA{id} = extraaxis;
            idxB{id} = extraaxis;
            sa0 = extra - a(idxA{:});
            sa0 = bsxfun( @rdivide, sa0, b(idxB{:}) );
            
        case 's' % (Basic DCD)            
            SimplifiedBasic = false;
            sa = extra;
    end

% 4 - Basic DCD (A = Aorth + Apar = Aorth + Sa * UNIT(B))
    if SimplifiedBasic
        Apar = bsxfun( @times, sa0, b );
        a = a + Apar;
        a(idxA{:}) = extra; % (removes rounding errors)
    else
        Apar = bsxfun( @times, sa, unit(b, id) );
        a = a + Apar;
    end



%--------------------------------------------------------------------------
function indices = ivector(sizeA)
%IVECTOR   Vectorizing the indices of an array. 

    Ndims = length(sizeA);
    indices = cell(1,Ndims); % preallocating
    for d = 1 : Ndims
       indices{d} = 1:sizeA(d);
    end
