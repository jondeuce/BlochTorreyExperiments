function setbtpath( type, varargin )
%SETBTPATH Set BlochTorreyExperiments and BlochTorreyResults path, e.g.
% setbtpath home/coopar7/master

if nargin == 0
    type = '.'; % use master branch in current folder
end

default_remove_list = ignoredfolders;

if nargin < 2
    remove_list = default_remove_list;
else
    remove_list = varargin;
end

switch upper(type)
    case 'UBCITAR' % ubcitar/jdoucette server path
        btroot = '/data/ubcitar/jdoucette/';
        btexp_branch = 'master';
        
    case 'UBCITAR/LAURA' % ubcitar/jdoucette server path
        btroot = '/data/ubcitar/jdoucette/';
        btexp_branch = 'Laura';
        
    case 'HOME/COOPAR7/MASTER' % coopar7 local path
        btroot = '/home/coopar7/Documents/code/';
        btexp_branch = 'master';
        
    case 'HOME/COOPAR7/TEMP1' % coopar7 local path
        btroot = '/home/coopar7/Documents/code/';
        btexp_branch = 'temp1';
        
    case 'ASUS/MASTER' % home asus laptop
        btroot = '/home/jon/Documents/UBCMRI/';
        btexp_branch = 'master';
        
    case 'ASUS/TEMP1' % home asus laptop
        btroot = '/home/jon/Documents/UBCMRI/';
        btexp_branch = 'temp1';
        
    case 'THINKPAD' % home thinkpad laptop
        btroot = 'C:\Users\Jonathan\Documents\MATLAB\';
        btexp_branch = 'master';
        
    case 'THINKPAD/TEMP1' % home thinkpad laptop
        btroot = 'C:\Users\Jonathan\Documents\MATLAB\';
        btexp_branch = 'temp1';
        
    case '.'
        btroot = [cd, '/'];
        btexp_branch = 'master';
        
    otherwise % assume type is a branch in the current directory
        btroot = [cd, '/'];
        btexp_branch = type;
end

BlochTorreyResults = 'BlochTorreyResults';
BlochTorreyExperiments = 'BlochTorreyExperiments';
if ~isempty(btexp_branch)
    BlochTorreyExperiments = [BlochTorreyExperiments, '-', btexp_branch];
end

% Remove folders
folders_to_remove = filterpath('BlochTorrey', path, true);
rmpath(folders_to_remove{:});

% Add folders
btexp_paths = filterpath({'.git', '+'}, genpath([btroot, BlochTorreyExperiments]), false);
btres_paths = filterpath({'.git', '+'}, genpath([btroot, BlochTorreyResults]), false);
for p = {btexp_paths, btres_paths}
    bt_paths = p{1};
    if ~isempty(bt_paths)
        addpath(bt_paths{:});
    end
end
cleanpath(remove_list{:});

cd(btroot);
savepath([btroot, 'btpathdef.m']);

end

function filtered_paths = filterpath(r, path_folders, remove)

if nargin < 3
    remove = true;
end

if nargin < 2
    path_folders = strsplit(path, ':');
elseif ischar(path_folders)
    path_folders = strsplit(path_folders, ':');
end

filter_indices = @(folds, r) ~cellfun(@isempty, strfind(folds, r));
if iscell(r)
    indices = filter_indices(path_folders, r{1});
    for ii = 2:numel(r)
        new_indices = filter_indices(path_folders, r{ii});
        indices = indices | new_indices;
    end
else
    indices = filter_indices(path_folders, r);
end

if ~remove
    indices = ~indices;
end

filtered_paths = path_folders(indices);

if remove
    filtered_paths = filtered_paths(~filter_indices(filtered_paths, matlabroot));
end

end

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
        paths_to_remove = genpath(remove_paths(ii).path);
        paths_to_remove = filterpath(matlabroot, paths_to_remove, false);
        rmpath(paths_to_remove{:});
    end
end

end
