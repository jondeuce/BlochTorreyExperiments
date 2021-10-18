function setbtpath
%SETBTPATH Set BlochTorreyExperiments path.

% Add folders
[btroot, ~, ~] = fileparts(mfilename('fullpath'));
addpath(genpath(btroot))
cleanpath(btroot)

end

function cleanpath(btroot)
%IGNOREDFOLDERS Filter folders that start with '.git', 'old', etc. from path

ignore_pattern = [btroot, '(?:[\s\S])*/(?:(?:.git)|(?:\+)|(?:old)|(?:Old)|(?:test)|(?:Test)|(?:testing)|(?:Testing)|(?:tmp)|(?:Temp)|(?:Tmp)|(?:backup)|(?:Backup)|(?:archive)|(?:Archive))(?:[\s\S])*'];
curr_folders   = split(path, ':');
folders_to_rm  = curr_folders(cellfun(@(x) ~isempty(regexp(x, ignore_pattern, 'once')), curr_folders));
rmpath(folders_to_rm{:});

end
