function [ Au ] = toeplitz_BT_fft( n, f, n_ind, level, func )
%TOEPLITZ_BT_FFT Direct assignment of Au

if nargin < 3 || ( isempty(n_ind) || isempty(level) )
    n_ind	=   ones(2,length(n));
    level	=   1;
end

if nargin < 5
    func	=   @(n,f,n_ind,level) reshape(1:f^2,f,f);
end

if level == length(n) + 1
    
    Au	=   flipud( func(n,f,n_ind,level) ).';
    Au	=   Au(:).';
    
else
    
    this_n	=	n(level);
    b_edge	=	f^2 * prod(2*n(level+1:end)-1);
        
    for ii = this_n:-1:1
        
        n_ind(1,level)	=   ii;
        blk             =   toeplitz_BT_fft(n,f,n_ind,level+1,func);
        Au(1+b_edge*(this_n-ii):b_edge*(this_n-ii+1))	=   blk;
        
    end
    
    for ii = 2:this_n
        
        n_ind(2,level)	=   ii;
        blk             =   toeplitz_BT_fft(n,f,n_ind,level+1,func);
        Au(1+b_edge*(ii+this_n-2):b_edge*(ii+this_n-1))	=   blk;
        
    end
    
end

end

