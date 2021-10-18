function testCROSS2
% TESTCROSS2  Testing function CROSS2
%    TESTCROSS2 performs a series of tests of function CROSS2

disp ' '
disp '--------------------------------------------------------------------'
disp '                       TESTING FUNCTION CROSS2'
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

[a, b, c] = crossAB([3 0], 1);
i = i+1; e(i) = check(a , b , c );
i = i+1; e(i) = check(a',-b ,-c');
i = i+1; e(i) = check(a ,-b',-c');
i = i+1; e(i) = check(a', b', c');

[a, b, c] = crossAB([3 1], 1);
i = i+1; e(i) = check(a , b , c );
i = i+1; e(i) = check(a',-b ,-c');
i = i+1; e(i) = check(a ,-b',-c');
i = i+1; e(i) = check(a', b', c');

[a, b, c] = crossAB([5 3], 2);
i = i+1; e(i) = check(a,-b,-c);

[a, b, c] = crossAB([3 1], 1);
[a2, b2, c2] = rearrange([1 1 1], [3 1], [], a, b, c);
i = i+1; e(i) = check(-a2, b2,-c2);
i = i+1; e(i) = check( a2,-b ,-c2);
i = i+1; e(i) = check( a,  b2, c2);

[a, b, c] = crossAB([2 3 7], 2);
[a2, b2, c2] = rearrange([1 1], [2 3 7], [], a, b, c);
i = i+1; e(i) = check( a , b , c , 2);
i = i+1; e(i) = check(-a2, b2,-c2, 4,4);
i = i+1; e(i) = check( a2, b , c2, 4,2);
i = i+1; e(i) = check( a ,-b2,-c2, 2,4);

[a, b, c] = crossAB([3 1], 1);
[a2, b2, c2] = rearrange([5 6], [3 1], [], a, b, c);
i = i+1; e(i) = check(a2,b ,c2, 3,1);
i = i+1; e(i) = check(a ,b2,c2, 1,3);

[a, b, c] = crossAB([1 1 2 3 7], 4);
[a2, b2, c2] = rearrange([1 1], [9 1 2 3 7], [], a, b, c);
i = i+1; e(i) = check(a2,b ,c2, 6,4);
i = i+1; e(i) = check(a ,b2,c2, 4,6);

[a, b, c] = crossAB([3 1], 1);
a = rearrange( 5, [3 1], [], a);
b = rearrange([], [3 2], [], b);
c = rearrange( 5, [3 2], [], c);
i = i+1; e(i) = check(a,b,c, 2,1);

[a, b, c] = crossAB([2 3], 2);
a = rearrange([1 5], [2 3], [], a);
b = rearrange([],    [2 3], 7,  b);
c = rearrange([1 5], [2 3], 7,  c);
i = i+1; e(i) = check(a,b,c);

disp ' '
disp ( ['Maximum error for all tests: ' num2str(max(e))] )
disp ( ['MATLAB precision:            ' num2str(eps)] )


function e = checkorth(a,b)
c0 = cross(a,b);
c = cross2(a,b);
disp ' '
fprintf ('               A = '), disp (a)
fprintf ('               B = '), disp (b)
fprintf ('    CROSS2(A, B) = '), disp (c)
e = max(abs(c-c0));


function [a, b, c] = crossAB(sizeAB, dim)
a = rand(sizeAB) - 0.5;
b = rand(sizeAB) - 0.5;
c = cross(a, b, dim);


function err = check(a, b, c0, varargin)

c = cross2(a, b, varargin{:});

% Setting IDA and/or IDB
switch nargin
    case 3
        idA = find(size(a)==3, 1, 'first'); % First dim. of length 3
        idB = find(size(b)==3, 1, 'first');
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
    error('CROSS2(A,B) and C are not the same size')
else
    err = c - c0;
    err = max( abs(err(:)) );
    disp (err)
end
disp '--------------------------------------------------------------------'

