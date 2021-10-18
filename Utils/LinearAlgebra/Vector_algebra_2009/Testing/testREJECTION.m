function testREJECTION
% TESTREJECTION  Testing function REJECTION
%    TESTREJECTION performs a series of tests of function REJECTION

disp ' '
disp '--------------------------------------------------------------------'
disp '                     TESTING FUNCTION REJECTION'
disp '            THE INTERNAL DIMENSIONS ARE INDICATED BY "•"'
disp '--------------------------------------------------------------------'

[a, b, aorth] = computeAB([3 1], 1);
e(1) = check(aorth,a,b);
e(2) = check(aorth',a',-b);
e(3) = check(aorth',a,-b');
e(4) = check(aorth',a',b');

    b = reshape(b,    [1 1 1 size(b)]); % 1x1x1x3
aorth = reshape(aorth,[1 1 1 size(aorth)]);
e(5) = check(aorth,a,b);

[a, b, aorth] = computeAB([3 1], 1);
    a = reshape(a,    [1 1 1 size(a)]); % 1x1x1x3
aorth = reshape(aorth,[1 1 1 size(aorth)]);
e(6) = check(aorth,a,-b);

[a, b, aorth] = computeAB([2 3 7], 2);
e(7) = check(aorth,a,b,2);

    a = reshape(a,    [1 1 1 size(a)]); % 1x1x2x3x7
aorth = reshape(aorth,[1 1 1 size(aorth)]);
e(8) = check(aorth,a,b,5,2);

[a, b, aorth] = computeAB([2 3 7], 2);
    b = reshape(b,    [1 1 1 size(b)]); % 1x1x2x3x7
aorth = reshape(aorth,[1 1 1 size(aorth)]);
e(9) = check(aorth,a,-b,2,5);

[a, b, aorth] = computeAB([3 1], 1);
    a = reshape(a,    [1 1 size(a)]); % 1x1x3
aorth = reshape(aorth,[1 1 size(aorth)]);
    a = repmat(a,     [5 6 1]); % 5x6x3
aorth = repmat(aorth, [5 6 1]);
e(10) = check(aorth,a,b,3,1);

[a, b, aorth] = computeAB([1 1 2 3 7], 4);
    b = reshape(b,    [1 1 size(b)]); % 1x1x1x1x2x3x7
aorth = reshape(aorth,[1 1 size(aorth)]);
    b = repmat(b,     [1 1 9 1 1 1 1]); % 1x1x9x1x2x3x7
aorth = repmat(aorth, [1 1 9 1 1 1 1]);
e(11) = check(aorth,a,b,4,6);

disp ' '
disp ( ['Maximum error for all tests: ' num2str(max(e))] )
disp ( ['MATLAB precision:            ' num2str(eps)] )


function [a, b, Aorth] = computeAB(size, dim)
 Apar = 10 * (rand(size) - 0.5);
    k = 10 * (rand(size) - 0.5);
Aorth = cross(Apar, k, dim);
    a = Apar + Aorth;
    b = (Apar) * -3.454;

function err = check(Aorth, a, b, varargin)

c = rejection(a, b, varargin{:});
% Setting IDA and/or IDB

switch nargin
    case 3
        idA0 = find(size(a)>1, 1, 'first'); % First non-singleton dim.
        idB0 = find(size(b)>1, 1, 'first'); % ([] if the array is a scalar)
        idA = max([idA0, 1]); % IDA = 1 if A is a scalar
        idB = max([idB0, 1]);
    case 4
        idA = varargin{1};
        idB = idA;
    case 5
        idA = varargin{1};
        idB = varargin{2};
end

idC = max(idA, idB); 

disp ' '
idstrA(10+idA*6) = '•';
idstrB(10+idB*6) = '•';
idstrC(10+idC*6) = '•';
diffA = idA - ndims(a);
diffB = idB - ndims(b);
diffC = idC - ndims(c);
disp (idstrA), fprintf ('Size of A:'), disp ([size(a) ones(1,diffA)])
disp (idstrB), fprintf ('Size of B:'), disp ([size(b) ones(1,diffB)])
disp (idstrC), fprintf ('Size of C:'), disp ([size(c) ones(1,diffC)])
disp ' '
err = c - Aorth;
err = max( abs(err(:)) );
fprintf ('Maximum error: ')
disp (err)
disp '---------------------------------------------------------------------'

