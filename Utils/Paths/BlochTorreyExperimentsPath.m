function [ p ] = BlochTorreyExperimentsPath
%BLOCHTORREYEXPERIMENTSPATH Returns the full path to the
%BlochTorreyExperiments folder, and returns an error if the folder is not
%found, or if there is more than one such folder.

p = finduniquefolder( 'BlochTorreyExperiments' );

end

