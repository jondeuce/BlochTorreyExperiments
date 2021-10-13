function [d] = fmg_max_depth(m)
    
    d = max(min(floor(log2(size(m)))) - 2, 1);
    
end
