function btpath( type, varargin )
%BTPATH Set BlochTorreyExperiments path.

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
        btroot = '/data/ubcitar/jdoucette/magprop_master';
        
    case 'HOME/COOPAR7/MASTER' % coopar7 local path
        btroot = '/home/coopar7/Documents/code/';
        btexp_branch = 'master';
        
    case 'ASUS' % home asus laptop
        error('not impl.');
        btroot = '/home/jon/Documents/UBCMRI';
        
    case 'THINKPAD' % home thinkpad laptop
        error('not impl.');
        btroot = 'C:\Users\Jonathan\Documents\MATLAB\magprop_master';
end

folders_to_remove = dir([btroot, 'BlochTorrey*']);
cleanpath(folders_to_remove.name);

addpath(genpath([btroot, '/BlochTorreyExperiments', '-', btexp_branch]));
addpath(genpath([btroot, '/BlochTorreyResults']));
cleanpath(remove_list{:});

cd(btroot);
savepath(which('pathdef'));

end

