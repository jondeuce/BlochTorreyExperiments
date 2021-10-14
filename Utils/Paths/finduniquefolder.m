function [ p ] = finduniquefolder( foldername )
%FINDUNIQUEFOLDER Returns the full path to the folder foldername if it
%exists and is unique. FINDUNIQUEFOLDER throws an error if the folder is
%not found, or if there is more than one such folder.

s = what( foldername );
switch numel(s)
    case 0
        error( 'folder "%s" not found!', foldername );
    case 1
        p = s.path;
    otherwise
        p = {s.path}.';
        error( ['Multiple folders "', foldername, '" found: %s\n'], p{:} );
end

end

