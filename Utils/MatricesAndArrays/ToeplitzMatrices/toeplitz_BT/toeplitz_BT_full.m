function [ A ] = toeplitz_BT_full( n, f, n_ind, level, func )
%TOEPLITZ_BT_FULL Algorithm 1: Generation of full $$T_{f}^{M}$$ MBT Matrices
% 
%	Recursive multilevel block Toeplitz (MBT) matrix generator
% 
%	n:      n = [ n1 n2 . . . nM ] is the number of BT blocks at each level
%   f:      f is the size of the final dense fxf block.
%   n_ind:  n_ind is size 2xn and indicates the current block index.
%               Row 1 => down, Row 2 => across
%   level:  current level (for recursive steps)
%
%   BT_full is initially called with only n and f as arguments

if nargin < 3 || ( isempty(n_ind) || isempty(level) )
    n_ind	=   ones(2,length(n));
    level	=   1;
end

if nargin < 5
    %func	=   @(n,f,n_ind,level) random_toeplitz_matrix(n,f,n_ind,level);
    func	=   @(n,f,n_ind,level) toeplitz_BT_index_func(n,f,n_ind,level);
end

if level == length(n) + 1
    
    %A	=	application_function(n,f,n_ind,level); % fxf block assignment
    A	=	func(n,f,n_ind,level); % fxf block assignment
    
else
    
    this_n	=   n(level);
    b_edge	=   prod( n(level+1:length(n)) ) * f;
    
    for ii = 1:this_n % lower triangle/diagonal assignment
        
        n_ind(1,level)	=	ii;
        blk             =   toeplitz_BT_full( n, f, n_ind, level+1, func );
        
        for jj = 1:(this_n-ii+1)
            
            A(  b_edge*(ii-1)+(b_edge*(jj-1)+1:b_edge*jj),	...
                b_edge*(jj-1)+1:b_edge*jj	)	=   blk;
            
        end
        
    end
    
    for ii = 2:this_n % upper triangle assignment
        
        n_ind(2,level)	=   ii;
        blk             =   toeplitz_BT_full( n, f, n_ind, level+1, func );
        
        for jj = 1:(this_n-ii+1)
            
            A(  b_edge*(jj-1)+1:b_edge*jj,	...
                b_edge*(ii-1)+(b_edge*(jj-1)+1:b_edge*jj)	)	=	blk;
            
        end
        
    end
    
end

end

function T = random_toeplitz_matrix(n,f,n_ind,level)

rand_complex_vec	=   @(n) complex( randn(n,1), randn(n,1) );

r       =   rand_complex_vec(f);
c       =   rand_complex_vec(f);
c(1)	=   r(1);

T       =   toeplitz(c,r);

end


