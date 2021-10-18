function [p, t] = circularmeshwithtori( bcircle, outer_centers, outer_radii, inner_centers, inner_radii, h0, eta, isunion, regiontype )
%CIRCULARMESHWITHTORI(bcircle, outer_centers, outer_radii, inner_centers, inner_radii, h0, eta, isunion, regiontype)
%      BCIRCLE:         Bounding circle [x0, y0, r0]
%      OUTER_CENTRES:   Outer circle centers (NCIRCLES x 2)
%      OUTER_RADII:     Inner circle radii   (NCIRCLES x 1)
%      INNER_CENTRES:   Outer circle centers (NCIRCLES x 2)
%      INNER_RADII:     Inner circle radii   (NCIRCLES x 1)
%      H0:              Initial edge length
%      ETA:             Parameter to shorten edges near circles. Zero gives none
%      ISUNION:         Flag to include circle interiors in domain (default true)
%      REGIONTYPE:      Flag indicating region type (0 to 4)

% if nargin == 0; runMinExample; return; end

[xc, yc, r] = deal(bcircle(1), bcircle(2), bcircle(3));
scale = 2*r;

if nargin < 9; regiontype = 0; end
if nargin < 8; isunion = true; end
if nargin < 7; eta = 5; end
if nargin < 6; h0 = scale/25; end

if isunion && ~(regiontype == 0); error('Input "regiontype" must equal 0 for "isunion" == true.'); end
if (regiontype > 1) && (length(outer_radii) > 1); error('Only one circle is allowed for annular or interior regions.'); end

if ~isunion && (regiontype == 0); regiontype = 1; end

issigned = ~isunion;
all_centers = cat(1, outer_centers, inner_centers);
all_radii = cat(1, outer_radii(:), inner_radii(:));

% Fixed points
nminpoints = 8; % circles should have at least 8 points
circlePoints = gencirclepoints(h0/2, all_centers, all_radii, nminpoints);
pfix = unique(circlePoints, 'rows');

% Distance functions
fdboundary = @(p) dcircle(p,xc,yc,r);
fdoutercircles = @(p) dcircles(p, outer_centers(:,1), outer_centers(:,2), outer_radii, issigned);
fdinnercircles = @(p) dcircles(p, inner_centers(:,1), inner_centers(:,2), inner_radii, issigned);
fcircles = @(p) dcircles(p, all_centers(:,1), all_centers(:,2), all_radii, issigned);
fclamp = @(d, dmin, dmax) max(min(d, dmax), dmin);

% Edge size functions
if eta == 0.0
    fh = @huniform;
else
    fh = @(p) h0 * (1 + eta * fclamp(abs(fcircles(p)), 0, scale/2)/scale);
end

% `regiontype` specific options
if isunion
    fd = fdboundary;
else
    switch regiontype
        case 1 % exterior
            fd = @(p) ddiff(fdboundary(p), fdoutercircles(p));
        case 2 % annulus
            pfix = pfix(isinoroncircle(pfix, outer_centers, outer_radii, sqrt(eps(scale))), :);
            fd = @(p) ddiff(dintersect(fdoutercircles(p), fdboundary(p)), fdinnercircles(p));
        case 3 % interior
            pfix = pfix(isinoroncircle(pfix, inner_centers, inner_radii, sqrt(eps(scale))), :);
            fd = @(p) dintersect(fdinnercircles(p), fdboundary(p));
        otherwise
            error('Region must be 1 (exterior), 2 (annular), or 3 (interior), but regiontype == %f', regiontype);
    end
end

% Generate mesh
bbox = [xc, yc; xc, yc] + 1.05 * [-r, -r; r, r];
[p, t] = distmesh2d(fd, fh, h0, bbox, pfix);

% Set boundary to be exact
deps = sqrt(eps)*h0;
radii = sqrt(p(:, 1).^2 + p(:, 2).^2);
bdry_point_inds = (r - deps <= radii) & (radii <= r + deps);
p(bdry_point_inds, 1) = r * (p(bdry_point_inds, 1) - xc) ./ radii(bdry_point_inds) + xc; % scale x-coords by r/radii
p(bdry_point_inds, 2) = r * (p(bdry_point_inds, 2) - yc) ./ radii(bdry_point_inds) + yc; % scale y-coords by r/radii

end

% function [p, t] = runMinExample
%
% % bbox    = [-20,-20; 20,20];
% % centers = [5.5, 5.5; -7.0,-0.8; -7.0,-0.8];
% % radii   = [     2.5;       4.9;       2.0];
%
% bbox    = [-1.0 -1.0;  1.0 1.0];
% centers = [ 0.0 -0.5; -0.4 0.8; 0.70 0.70];
% radii   = [      0.5;      0.5;      0.38];
%
% % scale = min(diff(bbox,1));
% % circlesPerRow = 3;
% % circlesPerCol = 3;
% % Ncircles = circlesPerRow * circlesPerCol;
% % [X,Y] = meshgrid(linspacePeriodic(bbox(1,1),bbox(2,1),circlesPerRow), ...
% %                  linspacePeriodic(bbox(1,2),bbox(2,2),circlesPerCol));
% % dX = (bbox(2,1) - bbox(1,1))/circlesPerRow;
% % dY = (bbox(2,2) - bbox(1,2))/circlesPerCol;
% % centers = [X(:), Y(:)];
% % radii = 0.5*min(dX,dY)/2*ones(Ncircles,1);
%
% Nmin = 100; % points for smallest circle
% h0 = 2*pi*min(radii)/Nmin; % approximate scale
% eta = 5; % approx ratio between largest/smallest edges
%
% figure
% [p, t] = squaremeshwithcircles( bbox, centers, radii, h0, eta );
%
% end
%
% % fd=@(p) drectangle(p,-1,1,-1,1);
% % fh=@(p) 0.03+0.3*abs(dcircles(p,[-0.2,0.3],[0,0],[0.25,0.5],false));
% % figure, [p,t]=distmesh2d(fd,fh,0.03,[-1,-1;1,1],[-1,-1;-1,1;1,-1;1,1]);
