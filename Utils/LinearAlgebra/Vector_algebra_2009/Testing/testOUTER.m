function testOUTER
% TESTOUTER  Testing function OUTER
%    TESTOUTER performs a series of tests of function OUTER

disp ' '
disp '--------------------------------------------------------------------'
disp '                       TESTING FUNCTION OUTER'
disp '            THE INTERNAL DIMENSIONS ARE INDICATED BY "•"'
disp '--------------------------------------------------------------------'
a = rand(5,1);
b = rand(3,1);
check(a,b);

a = rand(1,5);
b = rand(3,1);
check(a,b);

a = rand(5,1);
b = rand(1,3);
check(a,b);

a = rand(1,5);
b = rand(1,3);
check(a,b);

a = rand(5,1);
b = rand(1,1,1,3);
check(a,b);

a = rand(5,6,2);
b = rand(3,6,2);
c = check(a,b);

a = rand(5,6,2);
b = rand(3,1);
c = check(a,b,3,1);

a = rand(1,1,5,6,2);
b = rand(3,1);
c = check(a,b,5,1);

a = rand(1,1,5,6,2);
b = rand(3,6,2);
c = check(a,b,3,1); 

a = rand(1,1,5,6,2);
b = rand(1,1,1,9,1,3,6,2);
c = check(a,b,3,6);

function c = check(a, b, varargin)

c = outer(a, b, varargin{:});
% Setting IDA and/or IDB

% Setting IDA and/or IDB
switch nargin
    case 2
        idA0 = find(size(a)>1, 1, 'first'); % First non-singleton dim.
        idB0 = find(size(b)>1, 1, 'first'); % ([] if the array is a scalar)
        idA = max([idA0, 1]); % IDA = 1 if A is a scalar
        idB = max([idB0, 1]);
    case 3
        idA = varargin{1};
        idB = idA;
    case 4
        idA = varargin{1};
        idB = varargin{2};
end

idC(1) = max(idA, idB); 
idC(2) = idC + 1;

disp ' '
idstrA(10+idA*6) = '•';
idstrB(10+idB*6) = '•';
idstrC(10+idC*6) = '•';
diffA = idA    - ndims(a);
diffB = idB    - ndims(b);
diffC = idC(2) - ndims(c);
disp (idstrA), fprintf ('Size of A:'), disp ([size(a) ones(1,diffA)])
disp (idstrB), fprintf ('Size of B:'), disp ([size(b) ones(1,diffB)])
disp (idstrC), fprintf ('Size of C:'), disp ([size(c) ones(1,diffC)])
disp ' '
disp '---------------------------------------------------------------------'

