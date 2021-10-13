function [e,te,tn]= kmg2dtng(t)
%KMG2DTNG  Computes the triangle neighbor information
%--------------------------------------------------------------------
%   [e,te,tn]=kmg2dtng(t)
% Input:
%    t       : Triangle vertices nt*3 (3-node) or nt*6 (6-node)
% Output:
%    e       : Mesh edges ne*2 or ne*3; 
%              - 3-node e(i,1) and e(i,2) are endpoints edge i
%              - 6-node e(i,3) is midpoint of edge i
%   te       : Triangle egdes nt*3, te(i,k) is the edge opposite to 
%              node k in local numbering
%   tn       : Triangle neighbors nt*3, tn(i,k) is the triangle 
%              sharing the edge k (in local numbering) with i.
%              In case where a side of a triangle has no triangle 
%              (i.e. boundary edge) neighbor, a value 0 is assigned.
%--------------------------------------------------------------------
% (c) 2013, Koko J., ISIMA, koko@isima.fr
%--------------------------------------------------------------------
nt=size(t,1);

% Extract all edges
[e,~,je]=kmg2dedg(t);

% Form triangle sides
te=reshape(je,nt,3);

% Form triangle neighbors
ne=size(e,1);
it=[1:nt]'; tn=zeros(nt,3); 
T=sparse(it,te(:,1),1,nt,ne)+sparse(it,te(:,2),2,nt,ne)+sparse(it,te(:,3),3,nt,ne);
for i=1:ne
    Ti=T(:,i); ii=find(Ti); ni=length(ii);    
    if ni == 2
       tn(ii(1),Ti(ii(1)))=ii(2);
       tn(ii(2),Ti(ii(2)))=ii(1);
    end
end