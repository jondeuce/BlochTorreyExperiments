function [ ignored_folders ] = btFoldersIgnored( varargin )
%BTFOLDERSIGNORED Folders to remove from path (contain old, testing, or
%out-dated code).

ignored_folders = { ...
    'test', 'Test', 'testing', 'Testing', 'tmp', 'Temp', 'Tmp', ...
    'old', 'Old', 'backup', 'Backup', 'archive', 'Archive' ...
    };

if nargin > 0
    ignored_folders = [ ignored_folders, varargin{:} ];
end

end

