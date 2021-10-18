function proj=projectPlane(v,n,q)
%PROJECTPLANE projects the 3D cloud of vectors 'v' onto the plane defined
% by the 3-element normal vector n and the 3-element point q on the plane

if size(v,2)~=3
    if size(v,1)==3; v=v';
    else error('Vectors must be in 3D');
    end
end

n=n(:)';
n=n/norm(n);
q=q(:)';

basisPlane=null(n); %basis for the plane
basisCoefficients= bsxfun(@minus,v,q)*basisPlane;
proj=bsxfun(@plus,basisCoefficients*basisPlane.', q);

end