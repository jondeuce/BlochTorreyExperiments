function [e,ib,je]=kmg2dedg(t)
%KMG2DEDG Form all edges without duplication & extract boundary edges
%--------------------------------------------------------------------
%         [e]=kmg2dedg(t)
%      [e,ib]=kmg2dedg(t)
%   [e,ib,je]=kmg2dedg(t)
% Input:
%   t        : Triangle nodes nt*3 or nt*6
% Output:
%    e       : Mesh edges ne*2 (3-node) or ne*3 (6-node)
%               - ne*2; e(i,1) and e(i,2) are end points edge i
%               - ne*3, e(i,3) is the midpoint of [e(i,1), e(i,2)]
%   ib       : Boundary edges in e ( i.e. e(ib,:) are boundary edges)
%   je       : Index vector such that eh=e(je)
%              used in functions kmg2dref and kmg2dtng
%--------------------------------------------------------------------
% (c) 2013, Koko J., ISIMA, koko@isima.fr
%--------------------------------------------------------------------
%
np=max(max(t(:,1:3)));

% Extract all edges  
eh0=[t(:,[2 3]); t(:,[3 1]); t(:,[1 2])];
[eh,ih]=sort(eh0,2);

% Remove duplicates
ee=int64(eh(:,1)*(np+1)+eh(:,2));
[~,iee,jee]=unique(ee); 
e=eh(iee,:);

% Add edges midpoint if 6-node triangulation
if (size(t,2) > 3)
    eq=[t(:,4); t(:,5); t(:,6)];
    e=[e eq(iee)];
end

if (nargout == 1)
    return
end

% Extract boundary edges
ne=size(e,1); 
he=accumarray(jee,1,[ne 1]); ib=find(he==1);

if (nargout == 2)
    return
end

% Form edges middle points (used in mesh refinement)
je=jee;




