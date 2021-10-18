function [ b ] = cylinderIntersectionInBox( p1, v1, r1, box1, p2, v2, r2, box2 )
%CYLINDERINTERSECTIONINBOX Checks if cylinders intersect OR their bounded
%boxes intersect; this is a fast approximation to checking if the finite-
%length cylinders themselves intersect.

b	=	cylinderIntersection( p1, v1, r1, p2, v2, r2 );
if any(b)
    nb	=   sum(b);
    if size(box1,3) == 1 && size(box2,3) > 1
        box1	=   repmat(box1,1,1,nb);
        box2	=   box2(:,:,b);
    elseif size(box2,3) == 1 && size(box1,3) > 1
        box1	=   box1(:,:,b);
        box2	=   repmat(box2,1,1,nb);
    else
        box1	=   box1(:,:,b);
        box2	=   box2(:,:,b);
    end
    try
        b(b)	=	b(b) & OOBBIntersection( box1, box2 );
    catch me
        keyboard
    end
end

end

