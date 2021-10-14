function cleanpath( varargin )
%CLEANPATH Removes any folder + subfolders from path with name matching
% input arguments.

if nargin == 0
    return
end

folders_to_remove_from_path = varargin;

for f = folders_to_remove_from_path
    remove_paths = what(f{1});
    
    for ii = 1:length(remove_paths)
        rmpath(genpath(remove_paths(ii).path));
    end
end

end

