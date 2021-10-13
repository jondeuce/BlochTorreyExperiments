function testBAXFUN()
% testBAXFUN  Testing function BAXFUN 

disp ' ', disp ' ', disp 'Testing function BAXFUN'

disp ' '
disp 'Subtracting the column means from a matrix A'
a = magic(5); % .. 5×5
bax(a, mean(a)); % 5×5

disp 'Subtracting the matrix means from 10 matrices contained in A'
a = rand(3, 3, 10); % ............... 3×3×10
means = mean(reshape(a, 9, 10)); % .... 1×10
bax(a, means, 0, 1); % .............. 3×3×10

disp 'Comparing a single vector A with each of the 10 vectors contained in B'
a = [0, 0.5, 1]; % .. 1×3
b = rand(3, 10); % .... 3×10
bax(a, b, -1, 0); % ... 3×10

disp 'Comparing a single vector A with each of the 10 vectors contained in B'
a = rand(1, 3); % .... 1×(3)
b = rand(2, 5, 3); % 2×5×(3)
bax(a, b, 1); % .... 2×5×(3)

disp 'Multiplying matrix A by each element of B, i.e. multiplying each'
disp 'element of A by each element of B (all possible combinations)'
a = [1 2 3 4 5; 6 7 8 9 10]; % 2×5
b = [1 10 100]; % .............. 1×3
bax(a, b, 0, 1); % ........... 2×5×3

disp 'Multiplying matrix A by each element of B (B is a matrix), i.e.'
disp 'multiplying each element of A by each element of B (all possible combin.)'
a = rand(2, 5); %  2×5×1
b = rand(3, 4); %  ... 3×4
bax(a, b, 0, 2); % 2×5×3×4

disp 'Combining large arrays of different size'
a = rand(2, 10, 1, 5); % 2×10×1×5×1
b = rand(3, 1, 4); % ........ 3×1×4
bax(a, b, 0, 2); % ..... 2×10×3×5×4

% ERRORS
 a = rand(1, 2, 3, 4);
 b = rand(1,3);
% bax(a,b,3,[1 2]); % shiftB not a scalar
% bax(a,b,[],4);  % shiftA not a scalar
% bax(a,b,2.1,2); % not an integer
% bax(a,b,2,2.1); 
% bax(a,b,3,'a'); % Not numeric
% bax(a,b,'a',1);

disp ' ', disp 'Thank you'

function c = bax(a,b,varargin)
switch nargin
    case 2
        swapped = varargin;
    case 3
        swapped = [{0} varargin];
    case 4
        swapped = varargin([2 1]);
end
c  = baxdisp(a,b,varargin{:});
c2 = baxdisp(b,a,swapped{:});
if ~isequal(c,c2), 
    error('Unexpected different result after swapping A and B')
end

function c = baxdisp(a,b,varargin)
switch nargin
    case 2,    shiftA = 0;           shiftB = 0;
    case 3,    shiftA = varargin{1}; shiftB = 0;
    case 4,    shiftA = varargin{1}; shiftB = varargin{2};
end
shiftC = 0;

x = - min(shiftA, shiftB);
if x > 0
    shiftA = shiftA + x;
    shiftB = shiftB + x;
    shiftC = x;    
end

c = baxfun(@plus, a, b, varargin{:});

disp ' '
dispstr('A:', shiftA, size(a))
dispstr('B:', shiftB, size(b))
dispstr('C:', shiftC, size(c))
disp ' '
   
function dispstr(str1, shift, siz)

str2=' '; 
str2 = str2(1, ones(1,shift*6));

fprintf([str1 str2])
disp(siz)
