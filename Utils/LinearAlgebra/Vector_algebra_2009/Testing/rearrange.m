function varargout = rearrange(addlead, multiplesize, addtrail, varargin)
% REARRANGE  Modifies the size of arrays
%     A = REARRANGE(addlead, multiplesize, addtrail, A)
%     rearranges array A using RESHAPE and REPMAT.
%     [A, B, ...] = REARRANGE(addlead, multiplesize, addtrail, A, B, ...)
%     rearranges arrays A, B, ... using RESHAPE and REPMAT.
%     You can add leading and trailing dimensions of any length, and you
%     can use multiples of the existing dimensions.
for i = 1: length(varargin)
    a = varargin{i};
    sizeA = size(a);    
    a = reshape(a, [ones(1,length(addlead)) sizeA]);
    a2 = repmat(a, [addlead multiplesize./sizeA addtrail]);
    varargout{i} = a2;
end