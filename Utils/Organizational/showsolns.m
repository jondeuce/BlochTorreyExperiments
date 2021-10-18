function [ ] = showsolns( s )
%SHOWSOLNS Given struct of symbolic solution 's', print out results

for field = fields(s)'
    f = field{1};
    fprintf('%s: ',f);
    disp(s.(f));
end

end

