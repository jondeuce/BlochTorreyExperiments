function b = isRotationMatrix( m, tol )
%ISROTATIONMATRIX Checks if the input matrix 'm' is a valid rotation
%matrix. outputs a boolean value

if ~( ismatrix(m) && diff(size(m)) == 0 )
    b	=   false;
    return
end

if nargin < 2
    tol = 10*eps(class(m));
end

I	=   eye(size(m));
b	=	maxabs(m*m'-I) < tol	&&  ...
        abs(det(m)-1)  < tol;

end