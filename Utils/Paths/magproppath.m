function magproppath( type, varargin )
%MAGPROPPATH Switch between local path on coopar7 and network path on
%ubcitar.

if nargin == 0
    return
end

default_remove_list = { '.git', 'test', 'Test', 'testing', 'Testing', ...
    'old', 'Old', 'backup', 'Backup', 'test', 'ISMRM2018' };

if nargin < 2
    remove_list = default_remove_list;
else
    remove_list = varargin;
end

switch upper(type)
    case 'UBCITAR' % ubcitar/jdoucette server path
        magprop_root = '/data/ubcitar/jdoucette/magprop_master';
        
    case 'COOPAR7' % coopar7 local path
        magprop_root = '/home/coopar7/Documents/code/magprop_master';
        
    case 'COOPAR7/SHALLOW1' % coopar7 local path
        magprop_root = '/home/coopar7/Documents/code/magprop_shallow1';
        
    case 'ASUS' % home asus laptop
        magprop_root = '/home/jon/Documents/UBCMRI';
        
    case 'THINKPAD' % home thinkpad laptop
        magprop_root = 'C:\Users\Jonathan\Documents\MATLAB\magprop_master';
end

cleanpath('MagnetizationPropagation','MagPropCommon');
addpath(genpath([magprop_root, '/MagnetizationPropagation']));
addpath(genpath([magprop_root, '/MagPropCommon']));
cleanpath(remove_list{:});

cd(magprop_root); cd ..
savepath(which('pathdef'));

end

