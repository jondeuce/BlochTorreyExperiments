function path = parentfolder(depth)
%RELPATH Return absolute path given a path relative to the current working
%directory as given by pwd. Output is semantically equivalent to
%`normpath(pwd(), joinpath(varargin{:}))`.

if nargin < 1
    depth = 0;
end

if ischar(depth)
    if strcmp(depth, '.')
        depth = 0;
    elseif ~isempty(regexp(depth, '^\.\.(?:/\.\.)*$'))
        % '..' followed by zero or more '/..'
        depth = numel(split(depth, '/'));
    else
        error('Input must be an integer depth, or one of: ''.'', ''..'', ''../..'', etc.');
    end
elseif ~(isnumeric(depth) && numel(depth) == 1 && depth >= 0 && depth == round(depth))
    error('Input must be an integer depth, or one of: ''.'', ''..'', ''../..'', etc.');
end

path = pwd;
for ii = 1:depth
    [path, ~, ~] = fileparts(path);
end

end
