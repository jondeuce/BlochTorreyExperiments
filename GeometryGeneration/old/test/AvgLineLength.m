function [ output_args ] = AvgLineLength( input_args )
%AVGLINELENGTH Investigate the average line length of a random line through
%a cylinder

NumLines = 1e4;
NumBoxes = 1e3;

[phi,th,relX,relY,len] = deal(zeros(NumBoxes,1));
for ii = 1:NumBoxes
    % Uniformly random boxes (sorted s.t. side lengths are x < y < z)
    BoxDims  = sort(unit(abs(randn(1,3)),2));
    
    len(ii) = randLineLength( NumLines, BoxDims );
    
    % Box length in X/Y direction relative to Z direction (longest)
    relX(ii) = BoxDims(1)/BoxDims(3);
    relY(ii) = BoxDims(2)/BoxDims(3);
    
    % polar angles, using physics convention (not matlab's)
    [phi(ii), th(ii), ~] = cart2sph(BoxDims(1), BoxDims(2), BoxDims(3));
end

% figure, scatter3(phi, th, len);
% xlabel('$\phi$');
% ylabel('$\theta$');
% axis image

figure, scatter3(relX, relY, len);
xlabel('$X/Z$');
ylabel('$Y/Z$');
axis image

[Center,Radius] = sphereFit([relX, relY, len]);
spherePlot(Center, Radius, 1, 100);
% [center, radii, evecs, params ] = ellipsoidFit([relX, relY, len], 1);
% ellipsoidPlot(center, radii, 1);

xlim([min(relX), max(relX)]);
ylim([min(relY), max(relY)]);
zlim([min(len),  max(len)]);

% empirical results
[xc,yc,zc] = deal(1, 2/3, 0);
[xr,yr,zr] = deal(1, 4/3, 1/2);
z = @(relX, relY) zc + zr * sqrt(1 - min((relX-xc).^2 / xr^2 + (relY-yc).^2 / yr^2, 1));

end

function AvgLength = randLineLength( NumLines, BoxDims )

BoxCenter = zeros(size(BoxDims), 'like', BoxDims);

p = bsxfun(@times, rand(length(BoxDims), NumLines) - 0.5, BoxDims(:));
v = unit(randn(length(BoxDims), NumLines), 1);

% [~, ~, Lmin, Lmax] = rayBoxIntersection( p, v, BoxDims, BoxCenter );
% LineLengths = magn( Lmax - Lmin, 1 );

[tmin, tmax, ~, ~] = rayBoxIntersection( p, v, BoxDims, BoxCenter );
LineLengths = tmax - tmin;

try
    assert(sum(isnan(LineLengths)) == 0);
catch me
    keyboard
    rethrow(me);
end

% hist(LineLengths(:), min(1000, floor(NumLines/10)));
% title(['BoxDims = ', mat2str(BoxDims)]);

AvgLength = mean(LineLengths);

end