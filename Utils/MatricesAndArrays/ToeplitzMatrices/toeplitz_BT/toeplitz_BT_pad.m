function [ xz ] = toeplitz_BT_pad( n, f, x )
%TOEPLITZ_BT_PAD Generation of x_z by inserting zeros into x

xz	=   [];

if isempty(n)
    
    xz	=   [ x, zeros(1,f^2-f) ];
    
else
    
    b_edge	=   prod(n(2:end)) * f;
    this_n	=   n(1);
    
    for i = 1:this_n
        
        if length(n) > 1
            xz	=   [   xz, ...
                        toeplitz_BT_pad( n(2:end), f, x(1+(i-1)*b_edge:i*b_edge) ),  ...
                        zeros(1,(n(2)-1)*prod(2*(n(3:end))-1)*f^2)  ];
        else
            xz	=   [   xz, ...
                        toeplitz_BT_pad( n(2:end), f, x(1+(i-1)*b_edge:i*b_edge) )	];
        end
        
    end
    
end


end

