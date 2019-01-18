function setbtpath( type, varargin )
%SETBTPATH Set BlochTorreyExperiments and BlochTorreyResults path, e.g.
% setbtpath home/coopar7/master

if nargin == 0
    type = '.'; % use master branch in current folder
end

default_remove_list = btFoldersIgnored;

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

% Cannot add library directories to path (i.e. starting with "+").
% Also, want to ignore .git folders
genPath = @(s) genpath_exclude(s,{'*\.git*', '\+*'});

% remove folders
folders_to_remove = dir([btroot, 'BlochTorrey*']);
cleanpath(folders_to_remove.name);

% add folders
addpath(genPath([btroot, BlochTorreyExperiments]));
addpath(genPath([btroot, BlochTorreyResults]));
cleanpath(remove_list{:});

cd(btroot);
savepath([btroot, 'btpathdef.m']);

end

