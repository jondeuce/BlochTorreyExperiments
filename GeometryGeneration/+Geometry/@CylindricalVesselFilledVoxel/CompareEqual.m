function [ ] = CompareEqual( G, R )
%COMPAREEQUAL Compare two geometries to check if all of their fields are
% equal. Additionally, if a field is not exactly equal and they are numeric
% types, the maximum absolute difference is displayed. Else, inconsistent
% field is displayed.

for f = fieldnames(G).'
    g = G.(f{1});
    r = R.(f{1});
    
    try
        assert( isequal( g, r ) );
    catch
        if isnumeric( g ) && isnumeric( r )
            fprintf('%s maxabs error: %.16e\n', f{1}, maxabs(vec(g-r)));
        else
            warning('%s does not match', f{1});
        end        
    end
    
end

end

