function setbtpath( type, varargin )
%SETBTPATH Set BlochTorreyExperiments and BlochTorreyResults path, e.g.
% setbtpath home/coopar7/master

if nargin == 0
    return
end

default_remove_list = btFoldersIgnored;

if nargin < 2
    remove_list = default_remove_list;
else
    remove_list = varargin;
end

switch upper(type)
    case 'UBCITAR' % ubcitar/jdoucette server path
        error('not impl.');
        btroot = '/data/ubcitar/jdoucette/magprop_master/';
        
    case 'HOME/COOPAR7/MASTER' % coopar7 local path
        btroot = '/home/coopar7/Documents/code/';
        btexp_branch = 'master';
        
    case 'HOME/COOPAR7/TEMP1' % coopar7 local path
        btroot = '/home/coopar7/Documents/code/';
        btexp_branch = 'temp1';
        
    case 'ASUS' % home asus laptop
        error('not impl.');
        btroot = '/home/jon/Documents/UBCMRI/';
        
    case 'THINKPAD' % home thinkpad laptop
        btroot = 'C:\Users\Jonathan\Documents\MATLAB\';
        btexp_branch = 'master';
        
    case 'THINKPAD/TEMP1' % home thinkpad laptop
        btroot = 'C:\Users\Jonathan\Documents\MATLAB\';
        btexp_branch = 'TEMP1';
end

% Cannot add library directories to path (i.e. starting with "+").
% Also, want to ignore .git folders
genPath = @(s) genpath_exclude(s,{'*\.git*', '\+*'});

% remove folders
folders_to_remove = dir([btroot, 'BlochTorrey*']);
cleanpath(folders_to_remove.name);

% add folders
addpath(genPath([btroot, 'BlochTorreyExperiments', '-', btexp_branch]));
addpath(genPath([btroot, 'BlochTorreyResults']));
cleanpath(remove_list{:});

cd(btroot);
savepath([btroot, 'btpathdef.m']);

end

