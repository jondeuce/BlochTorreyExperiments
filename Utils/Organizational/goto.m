function goto( Destination, LevelsUp )
%GOTO Change folders to 'Destination'.
%
% INPUT ARGUMENTS
%   -'Destination':	A string containing either a folder name or a filename.
%                  	If it is a file, the current folder will be changed
%                   to the folder containing the file.
%   -'LevelsUp':  	A positive integer (may be a char) which indicates how
%                  	many levels up (successive parent folders) from the
%                   'Destination' folder to move up in the directory

if isa( Destination, 'function_handle' )
    Destination	=   func2str( Destination );
end

if nargin < 2
    LevelsUp = [];
elseif isa( LevelsUp, 'char' )
    LevelsUp = str2num( LevelsUp ); %#ok<ST2NM>
    if ~( LevelsUp == round( LevelsUp ) && LevelsUp > 0 )
        LevelsUp = [];
    end
end

fid = exist( Destination, 'file' );

switch fid
    case 2 % 'Destination' is an m-file
        s	=	which( Destination, '-all' );
        idx	=   get_user_choice( s, fid );
        cd( fileparts( s{idx} ) );
        
    case 3 % 'Destination' is a mex-file
        s	=	which( Destination, '-all' );
        idx	=   get_user_choice( s, fid );
        cd( fileparts( s{idx} ) );
        
    case 7 % 'Destination' is a directory
        s	=	what( Destination );
        s	=	{s.path}';
        idx	=   get_user_choice( s, fid );
        cd( s{idx} );
        
    otherwise
        return
end

if ~isempty( LevelsUp )
    for n = 1:LevelsUp; cd ..; end;
end

end

function idx = get_user_choice( s, fid )

switch fid
    case 2
        type	=   'file';
        types	=   'files';
    case 3
        type	=   'file';
        types	=   'files';
    case 7
        type	=   'directory';
        types	=   'directories';
end

if length(s) == 1
    idx	=   1;
else
    fprintf( '\nMultiple %s were found with that name:\n\n', types );
    for ii = 1:length(s)
        fprintf( '%d: %s\n', ii, s{ii} );
    end
    fprintf('\n');
    
    msg	=   sprintf( 'Which %s did you mean? [Number]: ', type );
    idx =   input( msg );
    fprintf('\n');
end

end

