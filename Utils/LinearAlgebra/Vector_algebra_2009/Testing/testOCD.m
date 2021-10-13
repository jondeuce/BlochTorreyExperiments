function testOCD
% TESTOCD  Testing function OCD
%    TESTOCD performs a series of tests of function OCD


disp ' '
disp '--------------------------------------------------------------------'
disp '                        TESTING FUNCTION OCD'
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

[Aorth, b, c] = crossAB([3 0], 1);
i = i+1; e(i) = check(c , b , Aorth );
i = i+1; e(i) = check(c',-b ,-Aorth');
i = i+1; e(i) = check(c ,-b',-Aorth');
i = i+1; e(i) = check(c', b', Aorth');

[Aorth, b, c] = crossAB([3 1], 1);
i = i+1; e(i) = check(c , b , Aorth );
i = i+1; e(i) = check(c',-b ,-Aorth');
i = i+1; e(i) = check(c ,-b',-Aorth');
i = i+1; e(i) = check(c', b', Aorth');

[Aorth, b, c] = crossAB([5 3], 2);
i = i+1; e(i) = check(c,-b,-Aorth);

[Aorth, b, c] = crossAB([3 1], 1);
[Aorth2, b2, c2] = rearrange([1 1 1], [3 1], [], Aorth, b, c);
i = i+1; e(i) = check(-c2, b2,-Aorth2);
i = i+1; e(i) = check( c2,-b ,-Aorth2);
i = i+1; e(i) = check( c,  b2, Aorth2);

[Aorth, b, c] = crossAB([2 3 7], 2);
[Aorth2, b2, c2] = rearrange([1 1], [2 3 7], [], Aorth, b, c);
i = i+1; e(i) = check( c , b , Aorth , 2);
i = i+1; e(i) = check(-c2, b2,-Aorth2, 4,4);
i = i+1; e(i) = check( c2, b , Aorth2, 4,2);
i = i+1; e(i) = check( c ,-b2,-Aorth2, 2,4);

[Aorth, b, c] = crossAB([3 1], 1);
[Aorth2, b2, c2] = rearrange([5 6], [3 1], [], Aorth, b, c);
i = i+1; e(i) = check(c2,b ,Aorth2, 3,1);
i = i+1; e(i) = check(c ,b2,Aorth2, 1,3);

[Aorth, b, c] = crossAB([1 1 2 3 7], 4);
[Aorth2, b2, c2] = rearrange([1 1], [9 1 2 3 7], [], Aorth, b, c);
i = i+1; e(i) = check(c2,b ,Aorth2, 6,4);
i = i+1; e(i) = check(c ,b2,Aorth2, 4,6);

[Aorth, b, c] = crossAB([3 1], 1);
    c = rearrange( 5, [3 1], [], c);
    b = rearrange([], [3 2], [], b);
Aorth = rearrange( 5, [3 2], [], Aorth);
i = i+1; e(i) = check(c,b,Aorth, 2,1);

[Aorth, b, c] = crossAB([2 3], 2);
    c = rearrange([1 5], [2 3], [], c);
    b = rearrange([],    [2 3], 7,  b);
Aorth = rearrange([1 5], [2 3], 7,  Aorth);
i = i+1; e(i) = check(c,b,Aorth);

disp ' '
disp ( ['Maximum error for all tests: ' num2str(max(e))] )
disp ( ['MATLAB precision:            ' num2str(eps)] )


function e = checkorth(a0,b)
c = cross(a0,b);
a = ocd(c, b);
disp ' '
fprintf ('            C = '), disp (c)
fprintf ('            B = '), disp (b)
fprintf ('    OCD(C, B) = '), disp (a)

e = max(abs(a-a0));


function [Aorth,b,c] = crossAB(sizeAB, dim)
a = rand(sizeAB) - 0.5;
b = rand(sizeAB) - 0.5;
c = cross(a, b, dim);
Apar = projection(a, b, dim);
Aorth = a - Apar;


function err = check(c, b, Aorth0, varargin)

Aorth = ocd(c, b, varargin{:});

% Setting IDC and/or IDB
switch nargin
    case 3
        idC = find(size(c)==3, 1, 'first'); % First dim. of length 3
        idB = find(size(b)==3, 1, 'first');
    case 4
        idC = varargin{1};
        idB = idC;
    case 5
        idC = varargin{1};
        idB = varargin{2};
end

idA = max(idC, idB);

disp ' '
idstrA(14+idA*6) = '•';
idstrB(14+idB*6) = '•';
idstrC(14+idC*6) = '•';
diffA = idA - ndims(Aorth);
diffB = idB - ndims(b);
diffC = idC - ndims(c);
disp (idstrC), fprintf ('Size of C:    '), disp ([size(c) ones(1,diffC)])
disp (idstrB), fprintf ('Size of B:    '), disp ([size(b) ones(1,diffB)])
disp (idstrA), fprintf ('Size of Aorth:'), disp ([size(Aorth) ones(1,diffA)])

disp ' '
fprintf ('Maximum error: ')

if isempty(Aorth) && isempty(Aorth0)
    err = 0;
    disp ('Empty matrix')
    disp ' '
elseif ~isequal(size(Aorth), size(Aorth0))
    error('OCD(C,B) and Aorth are not the same size')
else
    err = Aorth - Aorth0;
    err = max( abs(err(:)) );
    disp (err)
end
disp '--------------------------------------------------------------------'

