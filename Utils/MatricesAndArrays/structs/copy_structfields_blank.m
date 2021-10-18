function [s] = copy_structfields_blank(x,s)
%COPY_STRUCTFIELDS_BLANK Copies all fields from the structure 'x' to the
%structure 's' with empty contents.

for f = fieldnames(x).'
    [s.(f{1})]	=   deal([]);
end

end
