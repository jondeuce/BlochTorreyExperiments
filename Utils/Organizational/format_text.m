function new_msg = format_text( msg, line_width, break_char, is_centered )
%FORMAT_TEXT Formats the string msg to have maximum line length line_width,
%optionally appends with a string of break_char's, and optionally centers
%the text.

if nargin < 2 || isempty(line_width)
    line_width	=   75;
end

if nargin < 3 || isempty(break_char)
    break_char	=   '';
end

if nargin < 4 || isempty(is_centered)
    is_centered	=   false;
end

new_msg = format_paragraph(msg, line_width);

if is_centered
    for ii = 1:length(new_msg)
        ndiff       =   line_width - ( length(new_msg{ii}) - 2 );
        prefix      =   repmat( ' ', [1,ceil(ndiff/2)] );
        new_msg{ii}	=   [ prefix, new_msg{ii} ];
    end
end

if ~isempty(break_char) && isa(break_char,'char')
    nreps       =   ceil(line_width/numel(break_char));
    break_char	=   repmat( break_char(:)', [1,nreps] );
    break_char	=   break_char(1:line_width);
    break_char	=   strrep( break_char, '%', '%%' );
    break_char	=   [ break_char, '\n' ];
    new_msg     =   [ break_char, new_msg, break_char ];
end

end

function new_msg = format_paragraph(msg, line_width)

% max characters per line
if nargin < 2
    line_width = 75;
end

msg = strtrim(msg);
len = length(msg);
if len <= line_width
    new_msg = {[strtrim(msg), '\n']};
    return
end

ascii_space = 32; %space character
new_msg = cell( 1, ceil(len/line_width) );
line_num = 1;
while length(msg) > line_width
    seg = msg(1:line_width);
    idx = find( seg == ascii_space, true, 'last' );
    
    if ~isempty(idx)
        new_msg{line_num} = [seg(1:idx-1), '\n'];
        msg = msg(idx+1:end);
    else
        new_msg{line_num} = [seg, '\n'];
        msg = strtrim( msg(line_width+1:end) );
    end
    line_num = line_num + 1;
end

if line_num < length(new_msg), new_msg = new_msg(1:line_num); end
new_msg{line_num} = [strtrim(msg) '\n'];

end