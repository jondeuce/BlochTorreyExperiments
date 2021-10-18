function [ out ] = struct2arglist( s )
%STRUCT2ARGLIST [ out ] = struct2arglist( s )
% Returns a [1 x 2*NumFields] cell array of name-value pairs of the fields
% and values of the struct s. If s has length > 1, the first entry in s
% will be used.

out = zipper(fieldnames(s(1)).',struct2cell(s(1)).',2);

end

