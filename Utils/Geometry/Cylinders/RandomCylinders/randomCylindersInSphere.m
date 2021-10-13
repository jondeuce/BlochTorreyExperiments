function [ p, v, r ] = randomCylindersInSphere( ...
    SphereRadius, SphereCenter, RadiiFunction, Goal, ...
    AllowSelfIntersection, PlotCylinders )
%ADDINTERSECTINGCYLINDERS [ p, v, r ] = randomCylindersInSphere( SphereRadius, SphereCenter, RadiiFunction, Goal, AllowSelfIntersection, PlotCylinders )
% Create random cylinders within a sphere

% Input Parsing
if nargin < 6; PlotCylinders = false; end
if nargin < 5; AllowSelfIntersection = true; end
if ~AllowSelfIntersection
    warning('AllowSelfIntersection = false not implemented; defaulting to true');
    AllowSelfIntersection = true;
end

RelVolGoal = false;
NumCylGoal = false;
if 0 < Goal && Goal < 1
    RelVolGoal = true;
elseif Goal >= 1 && Goal == round(Goal)
    NumCylGoal = true;
else
    error('Goal must be a between 0 and 1 (volume fraction goal), or a whole number (cylinder number goal).');
end

SphereVol = 4/3 * pi * SphereRadius^3;
if NumCylGoal
    Chunk = Goal;
else
    r_mean = mean(vec(RadiiFunction(1, 100))); % approx mean radius
    Chunk = Goal * SphereVol / (pi * r_mean^2 * SphereRadius); % cylinder length is ~approx Rsphere on average
    Chunk = max(1, ceil(Chunk/10)); % make chunk size smaller, as it's just an approximation
end

p = zeros(3,0);
v = zeros(3,0);
r = zeros(1,0);
while true
    % Uniformly random points within a sphere (for p_; v_ is a random direction):
    % https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
    p_ = randn(3, Chunk);
    p_ = bsxfun(@times, p_, SphereRadius .* rand(1, Chunk).^(1/3) ./ sqrt(sum(p_.^2, 1)));
    v_ = randn(3, Chunk);
    v_ = bsxfun(@rdivide, v_, sqrt(sum(v_.^2, 1)));
    r_ = RadiiFunction(1, Chunk);
    
    p = cat(2, p, p_);
    v = cat(2, v, v_);
    r = cat(2, r, r_);
    
    if NumCylGoal
        break % Cylinders are all created in one chunk
    end
    
    if getVolume( p, v, r, SphereRadius, SphereCenter ) > Goal * SphereVol
        break % Approx. cylinder volume exceeds goal volume
    end
end

if PlotCylinders
    BoxDims = 2 * SphereRadius * ones(3,1);
    BoxCenter = SphereCenter;
    plot_cylinders_in_box( p, v, r, BoxDims, BoxCenter, sprintf('Cylinders: N = %d', length(r)) );
    drawnow;
end

end

function [Volume] = getVolume( p, v, r, SphereRadius, SphereCenter )

% Get the length of the cylinder axes within the sphere and return their
% approx volumes, determined by the length of the intersection of the axis
L = raySphereIntersectionLength( p, v, SphereRadius, SphereCenter );
Volume = sum(vec(pi .* r.^2 .* L));

end
