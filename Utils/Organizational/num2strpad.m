function [ s ] = num2strpad( x, n, type )
%NUM2STRPAD Converts the number x to a string and then pads it to the
% appropriate size.
% 
% INPUT ARGUMENTS:
%   x:      Number to be converted to a string
%   n:      Pad. If n is an integer, x is padded with zeros to length n
%           according to type. If n is a string, x is padded with the
%           elements of n.
%   type:   Type of padding. May be 'pre' or 'post'. Default is 'pre'.

if nargin < 3 || isempty(type), type = 'pre'; end
if isnumeric(n), n = repmat('0',1,n); else n = n(:).'; end

s	=   num2str(x);
d	=   length(s);
p	=   length(n);

switch upper(type)
    case 'POST'
        s	=   [s,n(end-p+d+1:end)];
    otherwise
        s	=   [n(1:p-d),s];
end

end

