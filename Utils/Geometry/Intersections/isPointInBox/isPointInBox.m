function [ b, d ] = isPointInBox( p, BoxCenter, BoxDims )
%ISPOINTINBOX Returns boolean list that is true if the corresponding point
% p is in the box, and false otherwise
% 
% INPUT ARGUMENTS
%   p:          [3x1 or 3xN] Array of points to check
%   BoxCenter:  [3x1 or 3xN] Locations of centers of boxes
%   BoxDims:    [3x1 or 3xN] Sidelengths of boxes
% 
% OUTPUT ARGUMENTS
%   b:          [1xN] Boolean array; true if point is in box, else false

[BoxCenter,BoxDims]	=   deal(BoxCenter(:),BoxDims(:));

if nargout > 1
    top	=   bsxfun( @minus, BoxCenter + BoxDims/2, p );
    bot	=   bsxfun( @minus, BoxCenter - BoxDims/2, p );
    
    d	=   min( abs(top), abs(bot) );
    d	=   min( d, [], 1 );
    
    top	=   (top >= 0);
    bot	=   (bot <= 0);
    b	=   top & bot;
    
    clear top bot
else
    b	=	bsxfun( @ge, BoxCenter + BoxDims/2, p )	&	...
            bsxfun( @le, BoxCenter - BoxDims/2, p );
end
b	=   all( b, 1 );

end

function testing

printpassed	=   @(str) fprintf( 'Test PASSED: %s\n', str );
printfailed	=   @(str) fprintf( 'Test FAILED: %s\n', str );

Ntrials     =	100;
Npoints     =   10000;
Nboxes      =   10000;
str         =   sprintf( ...
    '%d Trials, p [3x%d], BoxCenter [3x%d], BoxDims[3x%d]', ...
    Ntrials, Npoints, Nboxes, Nboxes );


% Expect TRUE:
b           =   false(1,Ntrials);
BoxCenter	=   10000 * (2*rand(3,Nboxes)-1);
BoxDims     =   10000 * (0.5 + rand(3,Nboxes)/2);
p           =   bsxfun( @plus,	BoxCenter, ...
                bsxfun( @times, BoxDims, (rand(3,Npoints)-0.5) ) );
for ii = 1:Ntrials
    b(ii)	=   all( isPointInBox( p, BoxCenter, BoxDims ) );
end

if all(b),	printpassed(['Expected TRUE, ' str]);
else        printfailed(['Expected TRUE, ' str]);
end

% Expect FALSE:
b           =   false(1,Ntrials);
BoxCenter	=   10000 * (2*rand(3,Nboxes)-1);
BoxDims     =   10000 * (0.5 + rand(3,Nboxes)/2);
x           =   rand(3,Npoints)-0.5;
x           =   x + 0.5*sign(x); %random numbers in (-1,-0.5)U(0.5,1)
p           =   bsxfun( @plus, BoxCenter, bsxfun( @times, BoxDims, x ) );
for ii = 1:Ntrials
    b(ii)	=   all( isPointInBox( p, BoxCenter, BoxDims ) );
end

if ~any(b),	printpassed(['Expected FALSE, ' str]);
else        printfailed(['Expected FALSE, ' str]);
end

end