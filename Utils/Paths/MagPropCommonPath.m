function [ p ] = MagPropCommonPath
%MAGPROPCOMMONPATH  Returns the full path to the MagPropCommon folder, and
%returns an error if the folder is not found, or if there is more than one
%such folder.

p = finduniquefolder( 'MagPropCommon' );

end

