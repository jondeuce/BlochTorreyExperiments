function [ b ] = cylinderIntersection( p1, v1, r1, p2, v2, r2 )
%CYLINDERINTERSECTION Outputs a boolean vector which is true when the
%cylinder [p1,v1,r1] intersects any of the cylinders [p2,v2,r2]

n1	=   size(p1,2);
n2	=	size(p2,2);

if n1 == 1 && n2 > 1
    n	=   n2;
    p1	=	repmat(p1,[1,n]);
    v1	=	repmat(v1,[1,n]);
elseif n2 == 1 && n1 > 1
    n	=   n1;
    p2	=	repmat(p2,[1,n]);
    v2	=	repmat(v2,[1,n]);
elseif n1 ~= n2
    error( 'Number of cylinders must either be equal or equal to 1' );
end

b	=	( skewLineDist( p1, p2, v1, v2 ) < r1 + r2 );

end

