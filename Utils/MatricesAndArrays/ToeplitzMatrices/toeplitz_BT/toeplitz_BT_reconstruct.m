function [ b ] = toeplitz_BT_reconstruct( n, f, bz )
%TOEPLITZ_BT_RECONSTRUCT Reconstruction of b from bz

b	=   [];

if isempty(n)
    
    b	=   bz(f:f:f^2);
    
else
    
    b_edge	=   prod(2*n(2:end)-1) * f^2;
    this_n	=   n(1);
    
    for ii = this_n:2*this_n-1
        
        b	=   [ b, toeplitz_BT_reconstruct( n(2:end), f, bz(1+(ii-1)*b_edge:ii*b_edge) ) ];
        
    end
    
end

end

