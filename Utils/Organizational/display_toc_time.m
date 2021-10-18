function time_strings = display_toc_time( t_secs, labels, addspace )
%DISPLAY_TOC_TIME Displayed time elapsed from output of 'toc' function

%==========================================================================
% Input parsing and setup
%==========================================================================

if nargin < 3 || isempty(addspace)
    addspacePre     =   0;
    addspacePost	=	0;
else
	[addspacePre, addspacePost]	=	handle_addspace( addspace );
end

N = numel(t_secs);

if nargin < 2
    labels = repmat({''},[1,N]);
end

if isa(labels,'char')
    labels = repmat( {labels}, [1,N] );
end

if (N > 1) && isa(labels,'cell') && ( numel(labels) == 1 )
    labels = repmat( labels, [1,N] );
end

clabels	=	char(labels);
labels	=	mat2cell( clabels, ones(1,size(clabels,1)), size(clabels,2) )';

%==========================================================================
% Display times
%==========================================================================

if addspacePre > 0;  fprintf( repmat( '\n', 1, addspacePre ) ); end

time_str = cell(size(labels));
for ii = 1:N
    time_str{ii}	=	display_single_toc_time( t_secs(ii), labels{ii} );
end

if addspacePost > 0; fprintf( repmat( '\n', 1, addspacePost ) ); end


if nargout > 0
    time_strings	=	time_str;
end

end

function time_str = display_single_toc_time( t_secs, label )


t_days = t_secs / 24 / 60 / 60;

if t_secs < 1
    str = sprintf( '%0.6f', t_secs );
    suffix = ' secs';
elseif t_secs < 60
    str = datestr( t_days, 'SS.FFF' );
    suffix = ' secs';
elseif t_secs < 60 * 60
    str = datestr( t_days, 'MM:SS.FFF' );
    suffix = ' mins';
elseif t_secs < 24 * 60 * 60
    str = datestr( t_days, 'HH:MM:SS.FFF' );
    suffix = ' hrs';
else
    str = datestr( t_days, 'DD:HH:MM:SS.FFF' );
    suffix = ' days';
end

if isempty(label)
    prefix = 'Elapsed time:';
else
    prefix = [ 'Elapsed time (' label '):' ];
end

str	=   [ prefix, '\t', str, suffix, '\n' ];
fprintf( str );

if nargout > 0
    time_str = str;
end

end

function [addspacePre,addspacePost] = handle_addspace( addspace )

if isscalar( addspace )
    if addspace < 0
        addspacePre     =  -addspace;
        addspacePost	=   0;
    else
        addspacePre     =   0;
        addspacePost	=   addspace;
    end
else
    addspacePre     =   abs(addspace(1));
    addspacePost	=   abs(addspace(2));
end

if ~( addspacePre == round(addspacePre) )
    warning( 'addspace must be an integer or a boolean; using default 0' );
    addspacePre     =   0;
end
if ~( addspacePost == round(addspacePost) )
    warning( 'addspace must be an integer or a boolean; using default 0' );
    addspacePost	=   0;
end

end