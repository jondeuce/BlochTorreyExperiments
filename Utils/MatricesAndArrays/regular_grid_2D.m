function [ x, y ] = regular_grid_2D( n, isrow )
%REGULAR_GRID_2D Outputs points in a regular grid on a [-0.5,0.5]^2 grid.
%The number of points must be between 1 and 9.

if ~( n == round(n) && ( ( 1 <= n && n <= 9 ) || ( n == round(sqrt(n))^2 ) ) )
    error('n must be a whole number in the range [1,...,9], or a square number.');
end

if nargin < 2
    isrow	=	false;
end

if n == round(sqrt(n))^2
    n2      =   round(sqrt(n));
    [x,y]	=	meshgrid( double( linspacePeriodic(-0.5,0.5,n2) ) );
    [x,y]	=   deal(x(:),y(:));
else
    switch n
        case 2
            x = linspacePeriodic(-0.5,0.5,2).';
            y = [0;0];
        case 3
            x = linspacePeriodic(-0.5,0.5,2).';
            x = [x(1);0;x(2)];
            y = linspacePeriodic(-0.5,0.5,2).';
            y = [y;y(1)];
        case 5
            z = (1/3)*exp(-1i*(2*pi/20))*exp(1i*linspacePeriodic(0,2*pi,5).');
            x = real(z);
            y = imag(z);
        case 6
            z = (1/3)*exp(-1i*(2*pi/20))*exp(1i*linspacePeriodic(0,2*pi,5).');
            x = [0;real(z)];
            y = [0;imag(z)];
        case 7
            z = (1/3)*exp(1i*linspacePeriodic(0,2*pi,6).');
            x = [0;real(z)];
            y = [0;imag(z)];
        case 8
            z = (-13/36)*exp(-1i*(2*pi/28))*exp(1i*linspacePeriodic(0,2*pi,7).');
            x = [0;real(z)];
            y = [0;imag(z)];
        otherwise
            error('n must be a whole number in the range [1,9] or be square');
    end
end

if isrow
    x	=   x.';
    y	=   y.';
end

end

