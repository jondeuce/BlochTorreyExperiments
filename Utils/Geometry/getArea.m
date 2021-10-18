function [A,d] = getArea(X,numLeaves)
%takes n x 2 point cloud "X" and partitions X into at least "numLeaves"
%axis-aligned bounded rectangles, gets the convex hull for each leaf, and
%sums the areas of each hull. Total area is returned in "A". Also returns
%the point densities in each leaf "d"

if nargin==1; numLeaves=40; end;

s=ceil(size(X,1)/numLeaves);
t=splitSet(tree(X),1,s);

A=0;
lvs=t.findleaves;
p=zeros(length(lvs),1);
count=1;
for ii=lvs
    P=t.get(ii);
    m=size(P,1);
    if m>=3
        vi=convhull(P(:,1),P(:,2));
        a=polyarea(P(vi,1),P(vi,2));
        p(count)=m/a;
        A=A+a;
        count=count+1;
    end
end

p=p(1:count-1,:);

if nargout>1; d=p; end;

end

function t=splitSet(t,n,s)

A=t.get(n);
if size(A,1)<=s
    return;
end

m=[ min(A,[],1); max(A,[],1) ];
t=t.set(n,m);

[~,i]=max(diff(m));
cutoff=mean(m);

i1=A(:,i)>=cutoff(i);
i2=A(:,i)<cutoff(i);

[t,n1]=t.addnode(n,A(i1,:));
[t,n2]=t.addnode(n,A(i2,:));

t=splitSet(t,n1,s);
t=splitSet(t,n2,s);

end
