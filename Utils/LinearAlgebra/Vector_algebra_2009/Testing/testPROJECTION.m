function testPROJECTION
% TESTPROJECTION  Testing function PROJECTION
%    TESTPROJECTION performs a series of tests of function PROJECTION

disp ' '
disp '--------------------------------------------------------------------'
disp '                     TESTING FUNCTION PROJECTION'
disp '            THE INTERNAL DIMENSIONS ARE INDICATED BY "•"'
disp '--------------------------------------------------------------------'


[a, b, apar] = computeAB([3 1], 1);
e(1) = check(apar,a,b);
e(2) = check(apar',a',-b);
e(3) = check(apar',a,-b');
e(4) = check(apar',a',b');

   b = reshape(b,   [1 1 1 size(b)]); % 1x1x1x3
apar = reshape(apar,[1 1 1 size(apar)]);
e(5) = check(apar,a,b);

[a, b, apar] = computeAB([3 1], 1);
   a = reshape(a,   [1 1 1 size(a)]); % 1x1x1x3
apar = reshape(apar,[1 1 1 size(apar)]);
e(6) = check(apar,a,-b);

[a, b, apar] = computeAB([2 3 7], 2);
e(7) = check(apar,a,b,2);

   a = reshape(a,   [1 1 1 size(a)]); % 1x1x2x3x7
apar = reshape(apar,[1 1 1 size(apar)]);
e(8) = check(apar,a,b,5,2);

[a, b, apar] = computeAB([2 3 7], 2);
   b = reshape(b,   [1 1 1 size(b)]); % 1x1x2x3x7
apar = reshape(apar,[1 1 1 size(apar)]);
e(9) = check(apar,a,-b,2,5);

[a, b, apar] = computeAB([3 1], 1);
   a = reshape(a,   [1 1 size(a)]); % 1x1x3
apar = reshape(apar,[1 1 size(apar)]);
   a = repmat(a,    [5 6 1]); % 5x6x3
apar = repmat(apar, [5 6 1]);
e(10) = check(apar,a,b,3,1);

[a, b, apar] = computeAB([1 1 2 3 7], 4);
   b = reshape(b,   [1 1 size(b)]); % 1x1x1x1x2x3x7
apar = reshape(apar,[1 1 size(apar)]);
   b = repmat(b,    [1 1 9 1 1 1 1]); % 1x1x9x1x2x3x7
apar = repmat(apar, [1 1 9 1 1 1 1]);
e(11) = check(apar,a,b,4,6);

disp ' '
disp ( ['Maximum error for all tests: ' num2str(max(e))] )
disp ( ['MATLAB precision:            ' num2str(eps)] )


function [a, b, Apar] = computeAB(size, dim)
 Apar = 10 * (rand(size) - 0.5);
    k = 10 * (rand(size) - 0.5);
Aorth = cross(Apar, k, dim);
    a = Apar + Aorth;
    b = (Apar) * -3.454;

function err = check(Apar, a, b, varargin)

c = projection(a, b, varargin{:});
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
err = c - Apar;
err = max( abs(err(:)) );
fprintf ('Maximum error: ')
disp (err)
disp '---------------------------------------------------------------------'

