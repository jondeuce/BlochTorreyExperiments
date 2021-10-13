function [ A ] = toeplitz_BT_index_func(n,f,n_ind,level,b)
%TOEPLITZ_BT_INDEX_FUNC Test function argument for creating a toeplitz matrix

if length(n) == 3 && all( diff(n) == 0 )
    
    if nargin < 5
        b	=   reshape( 1:prod(2*n-1), 2*n-1 );
    end
    
    b0	=   floor(size(b)/2)+1;
    m	=   ( n_ind(2,:) == 1 );
    sub	=   b0 - (n_ind(1,:)-1).*m + (n_ind(2,:)-1).*(~m);
    A	=   b( sub(3), sub(2), sub(1) ) * ones(f,f);
    
else
    
    %A	=   reshape(1:f^2,f,f);
    A	=   randi(prod(2*n-1),1,1);
    
end

end