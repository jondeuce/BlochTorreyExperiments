function testXDIV
% TESTXDIV  Testing function CROSSDIV
%    TESTXDIV performs a series of tests of function CROSSDIV

disp ' '
disp '------------------------------------------------------------------------------'
disp '                          TESTING FUNCTION CROSSDIV                        '
disp '                THE INTERNAL DIMENSIONS ARE INDICATED BY "•"'
disp '------------------------------------------------------------------------------'

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
disp '------------------------------------------------------------------------------'

[a,b,c, extras] = crossAB([3 0], 1);
extras2 = extras;
extras3 = extras;
for i = 1:5,
    extras2{i} = reshape(extras{i}, [0 1 0]);
    extras3{i} = extras{i}'; 
end
i = i+1; e(i) = check(c , b , a , extras);
i = i+1; e(i) = check(c', b , a', extras2);
i = i+1; e(i) = check(c , b', a', extras2);
i = i+1; e(i) = check(c', b', a', extras3);

[a,b,c, extras] = crossAB([3 1], 1);
i = i+1; e(i) = check(c , b , a , extras);
i = i+1; e(i) = check(c', b , a', extras);
i = i+1; e(i) = check(c , b', a', extras);
i = i+1; e(i) = check(c', b', a', extras);

[a,b,c, extras] = crossAB([5 3], 2);
i = i+1; e(i) = check(c, b, a, extras);

[a,b,c, extras] = crossAB([3 1], 1);
[a2, b2, c2] = rearrange([1 1 1], [3 1], [], a, b, c);
i = i+1; e(i) = check(c2, b2, a2, extras);
i = i+1; e(i) = check(c2, b , a2, extras);
i = i+1; e(i) = check(c,  b2, a2, extras);

[a,b,c, extras] = crossAB([2 3 7], 2);
[a2, b2, c2]   = rearrange([1 1], [2 3 7], [], a, b, c);
[extras2{1:5}] = rearrange([1 1], [2 1 7], [], extras{:});
i = i+1; e(i) = check(c,b,a, extras, 2);
i = i+1; e(i) = check(c2, b2, a2, extras2, 4,4);
i = i+1; e(i) = check(c2, b , a2, extras2, 4,2);
i = i+1; e(i) = check(c , b2, a2, extras2, 2,4);

[a,b,c, extras] = crossAB([3 1], 1);
[a2, b2, c2] = rearrange([5 6], [3 1], [], a, b, c);
 [extras{:}] = rearrange([5 6], [1 1], [], extras{:});
i = i+1; e(i) = check(c2,b ,a2, extras, 3,1);
i = i+1; e(i) = check(c ,b2,a2, extras, 1,3);

[a,b,c, extras] = crossAB([1 1 2 3 7], 4);
[a2, b2, c2] = rearrange([1 1], [9 1 2 3 7], [], a, b, c);
 [extras{:}] = rearrange([1 1], [9 1 2 1 7], [], extras{:});
i = i+1; e(i) = check(c2,b ,a2, extras, 6,4);
i = i+1; e(i) = check(c ,b2,a2, extras, 4,6);

[a,b,c, extras] = crossAB([3 1], 1);
          c = rearrange( 5, [3 1], [], c);
          b = rearrange([], [3 2], [], b);
          a = rearrange( 5, [3 2], [], a);
[extras{:}] = rearrange( 5, [1 2], [], extras{:});
i = i+1; e(i) = check(c,b,a, extras, 2,1);

[a,b,c, extras] = crossAB([2 3], 2);
          c = rearrange([1 5], [2 3], [], c);
          b = rearrange([],    [2 3], 7,  b);
          a = rearrange([1 5], [2 3], 7,  a);
[extras{:}] = rearrange([1 5], [2 1], 7,  extras{:});
i = i+1; e(i) = check(c,b,a, extras);

disp ' '
disp ( ['Maximum error for all tests: ' num2str(max(e))] )
disp ( ['MATLAB precision:            ' num2str(eps)] )


function e = checkorth(a,b)
c = cross(a,b);
a1 = crossdiv(c,b);
a2 = crossdiv('+t',c,b,pi/2);
disp ' '
fprintf ('                             C = '), disp (c)
fprintf ('                             B = '), disp (b)
fprintf ('    CROSSDIV(C, B) = OCD(C, B) = '), disp (a1)
fprintf ('    CROSSDIV(''+t'', C, B, pi/2) = '), disp (a2)
err(1,:) = a - a1;
err(2,:) = a - a2;
e = max( abs(err(:)) );


function [a,b,c, extras] = crossAB(sizeAB, dim)
a = rand(sizeAB) - 0.5;
b = rand(sizeAB) - 0.5;
c = cross(a, b, dim);

% CD coefficient
unitB = unit(b, dim);
sa = dot(unitB, a, dim);
% Theta
cosine = dot(a, b, dim) ./ ( magn(a,dim) .* magn(b,dim) );
theta = acos(cosine);
% Ax, Ay, Az
indices = ivector(a);
indices{dim} = 1; Ax = a(indices{:});
indices{dim} = 2; Ay = a(indices{:});
indices{dim} = 3; Az = a(indices{:});

extras = {sa, theta, Ax, Ay, Az}; % cell array


function err = check(c,b,a, extras, varargin)

% Setting IDC and/or IDB
switch nargin
    case 4
        idC = find(size(c)==3, 1, 'first'); % First dim. of length 3
        idB = find(size(b)==3, 1, 'first');
    case 5
        idC = varargin{1};
        idB = idC;
    case 6
        idC = varargin{1};
        idB = varargin{2};
end

idA = max(idC, idB); 
indicesC = ivector(c);
indicesB = ivector(b);

types = {'+s ' '+s   ' '+s'
         '+t ' '   +t' '+t'
         '+Ax' ' + ax' '+x'
         '+Ay' ' +ay ' '+y'
         '+Az' '+ az ' '+z'};
e(1) = compare(a, types,c,b,extras, varargin{:});
alltypes(:,1) = types(:,1);

indicesC{idC} = 1;
c2 = c; c2(indicesC{:}) = -1;
types = {'-x+s'
         '-x+t'
         '-x+x'
         '-x+y'
         '-x+z'};
e(2) = compare(a, types,c2,b,extras, varargin{:});
alltypes(:,2) = types(:,1);

indicesC{idC} = 2;
c2 = c; c2(indicesC{:}) = -1;
types = {'-Cy +s ' ' -y + s '
         '-Cy +t ' ' -y + t '
         '-cy +ax' ' -y + x'
         '-cy +ay' ' -y + y'
         '-cy +az' ' -y + z'};
e(3) = compare(a, types,c2,b,extras, varargin{:});
alltypes(:,3) = types(:,1);

indicesC{idC} = 3;
c2 = c; c2(indicesC{:}) = -1;
types = {'+ s -cz'
         '+ t -cz'
         '+Ax -Cz'
         '+Ay -Cz'
         '+ z -Cz'};
e(4) = compare(a, types,c2,b,extras, varargin{:});
alltypes(:,4) = types(:,1);

indicesB{idB} = 1;
b2 = b; b2(indicesB{:}) = -1;
types = {'+s-bx'
         '+t-bx'
         '+x-bx'
         '+y-bx'
         '+z-bx'};
e(5) = compare(a, types,c,b2,extras, varargin{:});
alltypes(:,5) = types(:,1);

indicesB{idB} = 2;
b2 = b; b2(indicesB{:}) = -1;
types = {'-By+s'
         '-By+t'
         '-By+x'
         '-By+y'
         '-By+z'};
e(6) = compare(a, types,c,b2,extras, varargin{:});
alltypes(:,6) = types(:,1);

indicesB{idB} = 3;
b2 = b; b2(indicesB{:}) = -1;
types = {'-bz+s'
         '-bz+t'
         '-bz+x'
         '-bz+y'
         '-bz+z'};
e(7) = compare(a, types,c,b2,extras, varargin{:});
alltypes(:,7) = types(:,1);


disp ' '
idstrA(10+idA*6) = '•';
idstrB(10+idB*6) = '•';
idstrC(10+idC*6) = '•';
diffA = idA - ndims(a);
diffB = idB - ndims(b);
diffC = idC - ndims(c);
disp (idstrC), fprintf ('Size of C:'), disp ([size(c) ones(1,diffC)])
disp (idstrB), fprintf ('Size of B:'), disp ([size(b) ones(1,diffB)])
disp (idstrA), fprintf ('Size of A:'), disp ([size(a) ones(1,diffA)])
disp ' '
disp ('Checked TYPEs of cross division:')
disp (alltypes)
err = max( abs(e(:)) );
fprintf ('Maximum error: ')
disp (err)
disp '------------------------------------------------------------------------------'


function maxerr = compare(a, types,c,b,extras,varargin)

err(5) = 0;
for row = 1:5
    for col = 1 : size(types, 2)
        Acomputed = crossdiv(types{row,col},c,b,extras{row}, varargin{:});
        if isempty(Acomputed) && isempty(a)
            err(row,col) = 0;
        elseif ~isequal(size(Acomputed), size(a))
            error('CROSSDIV(TYPE,C,B,EXTRA) and A are not the same size')
        else
            e = Acomputed - a;
            err(row,col) = max( abs(e(:)) );
        end
    end
end
maxerr = max(err(:));


function indices = ivector(a)
%IVECTOR   Vectorizing the indices of an array. 
%
%   INDICES = IVECTOR(A) is a cell array containing the vectorized
%   indices of A.

    siz = size(a);
    Ndims = length(siz);
    indices = cell(1,Ndims); % preallocating
    for d = 1 : Ndims
       indices{d} = 1:siz(d);
    end

