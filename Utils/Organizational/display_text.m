function new_msg = display_text( msg, line_width, break_char, is_centered, addspace )
%display_text( msg, line_width, break_char, is_centered, addspace )
% Displays the (potentially very long) string msg such that the
% maximum line length width is line_width. Optionally the message can be
% padded with strings of break_char's or be centered.
% 
% EXAMPLE:
% 
% >> display_text('WARNING: not saving results.', 50, '=')
% ==================================================
%            WARNING: not saving results.
% ==================================================

if nargin < 5 || isempty(addspace)
    addspace	=	[0,0];
elseif isscalar(addspace)
    addspace	=   round( addspace );
    if addspace < 0,    addspace	=   [ abs(addspace), 0 ];
    else                addspace	=   [ 0, addspace ];
    end
end
if nargin < 4 || isempty(is_centered)
    is_centered = true;
end
if nargin < 3 || isempty(break_char)
    break_char = '%';
end
if nargin < 2 || isempty(line_width)
    line_width = 75;
end

if ~iscell(msg); msg = {msg}; end

fprintf( repmat( '\n', 1, round(abs(addspace(1))) ) );

for ii = 1:numel(msg)
    msg_formatted	=	format_text( ...
        msg{ii}, line_width, break_char, is_centered );
    
    if ii == 1; fprintf(msg_formatted{1}); end
    
    for jj = 2:numel(msg_formatted)-1
        fprintf(msg_formatted{jj});
    end
    
    if ii == numel(msg); fprintf(msg_formatted{end}); end

end

fprintf( repmat( '\n', 1, round(abs(addspace(2))) ) );

if nargout > 0
    new_msg	=	msg;
end

end

