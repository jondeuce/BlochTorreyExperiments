function th = averageAngle( angles, weights )
%AVERAGEANGLE computes the average of the angles in 'angles'

if nargin < 2
    weights = ones(size(angles));
end

th = complexAverage(angles, weights);

end

function th = complexAverage(angles, weights)

W = sum(weights);
th = atan2( sum(sin(angles).*weights)/W, sum(cos(angles).*weights)/W );

end

function th = modularAverage(angles, weights)


end