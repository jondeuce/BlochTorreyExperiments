function testUNIT
% TESTUNIT  Testing function UNIT
%    TESTUNIT performs a series of tests of function UNIT

format compact;

% Empty arrays
A1 = [];            
A2 = rand(3,0);     
A3 = A2';
A4 = rand(5,0,0,0); 
A5 = rand(0,0,0,5); 

% Matrices (2D arrays)
A6 = 0;                
A7 = [0 0 0];           
A8 = A7';
A9 = [0 0 3];           
A10 = A9';
A11 = [0 0 3
       0 0 4
       0 0 5];           
A12 = A11';
A13 = [0 0 3
       0 0 0
       0 0 0];           
A14 = A13';
A15 = [4 0 3
       0 0 0
       3 0 4];           
A16 = rand(3,3) * 1e15;   
A17 = rand(1,6) * 1e-15;  
A18 = A17';
A19 = rand(6,6) * 1e15;   

% Multidimensional arrays
A20 = rand(1,1,3);         
A21 = rand(1,1,3,1,2);  
A22 = rand(4,1,3,1,2);  

for i=1:22
    eval(['test(A' int2str(i) ')']);
end
for i=1:22
    eval(['testDIM(A' int2str(i) ')']);
end

function test(A)
disp ' '
disp '--------------------------------------------------------------------------'
disp '                          TESTING FUNCTION UNIT'
disp 'UnitA: normalized A'
disp 'MagnU: magnitude of UnitA' 
disp '--------------------------------------------------------------------------'
disp ' '
sizeA = size(A);
fprintf('Size of A:  '),
fprintf(1, '%0.0f', sizeA(1)), 
fprintf(1, ' x %0.0f', sizeA(2:end)), 
disp ' ', disp ' '
A
disp ' '
UnitA = unit(A)
disp ' '
MagnU = magn(UnitA)
disp ' '
if any( (abs(MagnU(:)-1)) > 2*eps)
    disp ':-('
    disp ':-(   There''s some vector in UnitA with magnitude not equal to 1'
    disp ':-('    
    disp 'NOTE: Due to a bug in the builtin function SUM, the magnitude of "vectors"'
    disp '      contained in empty arrays might not be an empty array, as expected.'
else
    disp ':-)'    
    disp ':-)   All vectors in UnitA have magnitude 1 (or NaN)'
    disp ':-)'    
end
disp ' '
disp 'Press any key to continue'
pause
disp ' '

function testDIM(A)
disp ' '
disp '**********************************************************************'
disp '                         TESTING FUNCTION UNIT                        '
disp '                                PHASE 2                               '
disp '                 ------> (USING PARAMETER DIM) <------                '
disp 'UnitA: normalized A'
disp 'MagnU: magnitude of UnitA' 
disp '**********************************************************************'
disp ' '

for DIM = 1 : ndims(A)+1
    disp ' '
    disp '--------------------'
    fprintf('DIM: '),
    disp (DIM)
    disp '--------------------'
    disp ' '
    sizeA = size(A);
    fprintf('Size of A:  '),
    fprintf(1, '%0.0f', sizeA(1)), 
    fprintf(1, ' x %0.0f', sizeA(2:end)), 
    disp ' ', disp ' '
    A
    disp ' '
    UnitA = unit(A, DIM)
    disp ' '
    MagnU = magn(UnitA, DIM)
    disp ' '
    if any( (abs(MagnU(:)-1)) > 2*eps)
        disp ':-('
        disp ':-(   There''s some vector in UnitA with magnitude not equal to 1'
        disp ':-('
        disp 'NOTE: Due to a bug in the builtin function SUM, the magnitude of "vectors"'
        disp '      contained in empty arrays might not be an empty array, as expected.'
    else
        disp ':-)'
        disp ':-)   All vectors in UnitA have magnitude 1 (or NaN)'
        disp ':-)'
    end
    disp ' '
    disp 'Press any key to continue'
    pause
end
disp ' '

        