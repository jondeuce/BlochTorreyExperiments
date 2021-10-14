function [path] = absfolderpath(varargin)
%ABSFOLDERPATH Get absolute path to folder relative to current directory.
%NOTE: folder must exist.

relpath = fullfile(varargin{:});
path = absdir(relpath);

if ~exist(path, 'dir')
    error('Directory does not exist: ''%s''', relpath);
end

end

function [path] = absdir(path)

if isunix
    [~, path] = system(['cd ', path, '; pwd']);
    path = strip(path);
end

end
