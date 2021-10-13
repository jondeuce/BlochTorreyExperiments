function [ h ] = plotOOBB( p, varargin )
%PLOTOOBB Plots the OOBB p.
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

faces	=   { [1,2,3,4], [1,2,6,5], [2,3,7,6], [3,4,8,7], [4,1,5,8], [5,6,7,8] };

hh	=   [];
for kk = 1:size(p,3)
    for ii = 1:6
        hh	=   [ hh; patch(p(1,faces{ii},kk)', p(2,faces{ii},kk)', p(3,faces{ii},kk)', ones(4,1)) ];
        hold on, axis image
    end
end

if nargin < 2; args = {'facecolor','b','edgecolor','k','facealpha',0.2}; end
set(hh,args{:});

if nargout > 0
    h	=   hh;
end

end

