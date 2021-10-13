function testDOT2
% TESTDOT2  Testing function DOT2
%    TESTDOT2 performs a series of tests of function DOT2

disp ' '
disp '--------------------------------------------------------------------'
disp '                       TESTING FUNCTION DOT2'
disp '            THE INTERNAL DIMENSIONS ARE INDICATED BY "•"'
disp '--------------------------------------------------------------------'

i = 1;
disp ' '
disp 'Orthogonal unit vectors'
a = [1 0 0];
b = [0 1 0];
i = i+1; e(i) = checkorth(a, b);
a = [0 1 0];
b = [0 0 1];
i = i+1; e(i) = checkorth(a, b);
a = [0 0 1];
b = [1 0 0];
i = i+1; e(i) = checkorth(a, b);
disp '--------------------------------------------------------------------'

[a, b, c] = dotAB([3 0], 1);
i = i+1; e(i) = check(a , b , c );
i = i+1; e(i) = check(a',-b ,-c');
i = i+1; e(i) = check(a ,-b',-c');
i = i+1; e(i) = check(a', b', c');

[a, b, c] = dotAB([3 1], 1);
i = i+1; e(i) = check(a , b , c );
i = i+1; e(i) = check(a',-b ,-c');
i = i+1; e(i) = check(a ,-b',-c');
i = i+1; e(i) = check(a', b', c');

[a, b, c] = dotAB([5 3], 2);
i = i+1; e(i) = check(a,-b,-c, 2);

[a, b, c] = dotAB([3 1], 1);
[a2, b2] = rearrange([1 1 1], [3 1], [], a, b); % C is a scalar
i = i+1; e(i) = check(-a2, b2,-c);
i = i+1; e(i) = check( a2,-b ,-c);
i = i+1; e(i) = check( a,  b2, c);

[a, b, c] = dotAB([2 3 7], 2);
[a2, b2] = rearrange([1 1], [2 3 7], [], a, b);
      c2 = reshape(c, [1 1 2 1 7]);
i = i+1; e(i) = check( a , b , c , 2);
i = i+1; e(i) = check(-a2, b2,-c2, 4,4);
i = i+1; e(i) = check( a2, b,  c2, 4,2);
i = i+1; e(i) = check( a ,-b2,-c2, 2,4);

[a, b, c] = dotAB([3 1], 1);
[a2, b2] = rearrange([5 6], [3 1], [], a, b);
      c2 = rearrange([5 6], [1 1], [], c);
i = i+1; e(i) = check(a2,b ,c2, 3,1);
i = i+1; e(i) = check(a ,b2,c2, 1,3);

[a, b, c] = dotAB([1 1 2 3 7], 4);
[a2, b2] = rearrange([1 1], [9 1 2 3 7], [], a, b);
      c2 = rearrange([1 1], [9 1 2 1 7], [], c);
i = i+1; e(i) = check(a2,b ,c2, 6,4);
i = i+1; e(i) = check(a ,b2,c2, 4,6);

[a, b, c] = dotAB([3 1], 1);
a = rearrange( 5, [3 1], [], a);
b = rearrange([], [3 2], [], b);
c = rearrange( 5, [1 2], [], c);
i = i+1; e(i) = check(a,b,c, 2,1);

[a, b, c] = dotAB([2 3], 2);
a = rearrange([1 5], [2 3], [], a);
b = rearrange([],    [2 3], 7,  b);
c = rearrange([1 5], [2 1], 7,  c);
i = i+1; e(i) = check(a,b,c, 4,2);

disp ' '
disp ( ['Maximum error for all tests: ' num2str(max(e))] )
disp ( ['MATLAB precision:            ' num2str(eps)] )


function e = checkorth(a,b)
c0 = dot(a,b);
c = dot2(a,b);
disp ' '
fprintf ('             A = '), disp (a)
fprintf ('             B = '), disp (b)
fprintf ('    DOT2(A, B) = '), disp (c)
e = max(abs(c-c0));


function [a, b, c] = dotAB(sizeAB, dim)
a = (rand(sizeAB) - 0.5);
b = (rand(sizeAB) - 0.5);
c = dot(a, b, dim);


function err = check(a, b, c0, varargin)

c = dot2(a, b, varargin{:});
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
fprintf ('Maximum error: ')

if isempty(c) && isempty(c0)
    err = 0;
    disp ('Empty matrix')
    disp ' '
elseif ~isequal(size(c), size(c0))
    error('DOT2(A,B) and C are not the same size')
else
    err = c - c0;
    err = max( abs(err(:)) );
    disp (err)
end
disp '--------------------------------------------------------------------'

