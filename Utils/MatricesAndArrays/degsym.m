function [ deg ] = degsym( varargin )
%DEGSYM Returns the degree symbol '°', a.k.a. char(176)

if nargin == 0
    deg = char(176);
else
    deg = repmat(char(176),varargin{:});
end

end

