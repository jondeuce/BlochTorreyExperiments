function [ b ] = OOBBIntersection( box1, box2 )
%OOBBINTERSECTION Checks if OOBB's p1 and p2 intersect. p1 and p2 must be
%[3x8xN] arrays where each column specifies a corner of an OOBB as follows:
% 
%                  8 ________________ 7
%                   /|              /|
%                  / |             / |
%                 /  |            /  |
%                /   |           /   |
%               /    |          /    |
%              /   4 |_________/_____| 3
%             /     /         /     /
%          5 /_____/_________/ 6   /
%            |    /          |    /
%            |   /           |   /
%            |  /            |  /
%            | /             | /
%            |/______________|/
%          1                   2

if size(box1,3) == 1 && size(box2,3) > 1
    box1	=   repmat(box1,1,1,size(box2,3));
elseif size(box2,3) == 1 && size(box1,3) > 1
    box2	=   repmat(box2,1,1,size(box1,3));
end

n	=   get_normals(box1,box2);
b	=	check_intersection( box1, box2, n );

end

function n = get_normals(box1,box2)

n	=   cell(1,size(box1,3));
for kk = 1:size(box1,3)
    n1	=	[	box1(:,2,kk) - box1(:,1,kk),	...
                box1(:,4,kk) - box1(:,1,kk),    ...
                box1(:,5,kk) - box1(:,1,kk)     ];
    n2	=   [	box2(:,2,kk) - box2(:,1,kk),    ...
                box2(:,4,kk) - box2(:,1,kk),    ...
                box2(:,5,kk) - box2(:,1,kk)     ];
    
    for ii = 1:3
        b	=   true(1,size(n2,2));
        for jj = 1:size(n2,2)
            b(jj)	=   b(jj) && norm(cross(n1(:,ii),n2(:,jj))) > 5 * eps(class(box1));
        end
        n2	=   n2(:,b);
        if isempty(n2)
            break
        end
    end
    
    if isempty(n2)
        n{kk}	=   n1;
        continue
    end
    
    n3	=   zeros(3,size(n1,2)*size(n2,2));
    ix	=   0;
    for ii = 1:3
        for jj = 1:size(n2,2)
            ix          =   ix + 1;
            n3(:,ix)	=   cross(n1(:,ii),n2(:,jj));
        end
    end
    
    n{kk}	=   unit([n1,n2,n3],1);
end

end

function b = check_intersection( box1, box2, n )

b	=   true(1,length(n));
for kk = 1:length(n)
    for jj = 1:size(n{kk},2)
        t1	=   sum( bsxfun(@times, box1(:,:,kk), n{kk}(:,jj)), 1 );
        t2	=   sum( bsxfun(@times, box2(:,:,kk), n{kk}(:,jj)), 1 );
        b(kk)	=   b(kk)	&	bsxfun( @le, min(t1,[],2), max(t2,[],2) )	...
                            &	bsxfun( @ge, max(t1,[],2), min(t2,[],2) );
        if ~b(kk)
            break
        end
    end
end

end