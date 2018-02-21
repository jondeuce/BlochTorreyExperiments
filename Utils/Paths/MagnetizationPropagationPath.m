function [ p ] = MagnetizationPropagationPath
%MAGNETIZATIONPROPAGATIONPATH Returns the full path to the
%MagnetizationPropagation folder, and returns an error if the folder is not
%found, or if there is more than one such folder.

p = finduniquefolder( 'MagnetizationPropagation' );

end

