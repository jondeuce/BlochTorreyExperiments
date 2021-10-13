function backupFile( varargin )
%BACKUPFILE Backs up a file to one or multiple specified locations
% 
% backupFile(SourceFileName, BackupLocations, AllowOverwrite) copies the
% file 'SourceFileName' to the locations specified by 'BackupLocations',
% overwriting existing files of the same name only if 'AllowOverwrite' has
% the value true.
% 
% backupFile(SourceFileName, BackupLocations, NewFileName, AllowOverwrite)
% has the same functionality as above, with the exception that the source
% file is renamed to 'NewFileName' after being copied. Both files given by
% 'SourceFileName' and 'NewFileName' must not already exist if the source 
% file is to be copied to the new location and 'AllowOverwrite' is false.
% 
% INPUT PARAMETERS:
%	SourceFileName:     1 x n string defining the source file or directory.
%	BackupLocations:	1 x n string or cell array of strings defining
%                       destination directory(s).
%	AllowOverwrite:     boolean which dictates whether or not existing
%                       backed up files may be overwritten.
%	NewFileName:        (optional) 1 x n string defining the new name of
%                       the file being backed up.

%% Parse arguments

if nargin < 3
    error( ['ERROR (' mfilename '): Minimum of 3 arguments required.'] );
elseif nargin == 3
    SourceFileName	=	varargin{1};
    BackupLocations	=	varargin{2};
    NewFileName     =	SourceFileName;
    AllowOverwrite	=	varargin{3};
    RenameFile      =   false;
elseif nargin == 4
    SourceFileName	=	varargin{1};
    BackupLocations	=	varargin{2};
    NewFileName     =	varargin{3};
    AllowOverwrite	=	varargin{4};
    RenameFile      =   true;
else
    error( ['ERROR (' mfilename '): Too many input arguments.'] );
end

if ischar( BackupLocations )
    BackupLocations	=   { BackupLocations };
end

%% Backup Files

[~,	SourceName, SourceExt] = fileparts(which(SourceFileName));
NewFileShortName = [ NewFileName, SourceExt ];

for BackupLocation = BackupLocations
    BackupThisFile      =	false;
    BackupFileFullName	=   [ BackupLocation{1}, NewFileShortName ];
    
    if ~exist( BackupFileFullName, 'file' )
        BackupThisFile	=	true;
    elseif AllowOverwrite
        warning( ...
            [	'Backup file for ' which( SourceName ) ' ' ...
                'already exists in ' BackupLocation{1} ' ' ...
                'and is being allowed to be overwritten.' ] );
        BackupThisFile	=	true;
    end
    
    if BackupThisFile
        % Copy file to new location
        copyfile( which( SourceName ), BackupLocation{1} );
        
        % Rename file if specified
        if RenameFile
            movefile( which( SourceName ), BackupFileFullName );
        end
    end
    
end

end

