function test_BlochTorreyAction
%TEST_BLOCHTORREYACTION Tests the function BlochTorreyAction

n = 25;
main_test([num2str(n),' 3D complex double'],n,100,[1,3]);
main_test([num2str(n),' 4D complex double'],n,30,[1,4]);

end

function main_test(label,n,varargin)

test_time = tic;

for ii = 1:n
    x  = randnc(10+randi(varargin{:}),'double');
    G  = randnc(size(x),'double');
    h  = rand;
    D  = rand;
    f  = -6*D/h^2 - G;
    iters = randi(3);
    
    gsize = size(x);
    ndim  = ndims(x);
    
    z  = x;
    if randi(2) == 1 %randomly reshape x to (repeated-)column vector
        if ndim == 3; z = z(:);
        else z = reshape(z,[prod(gsize(1:3)),gsize(4)]);
        end
    end
    
    y1 = BlochTorreyAction(z,h,D,f,gsize(1:3),iters);
    y1 = reshape(y1,size(x));
    y2 = BlochTorreyAction_brute(x,h,D,f,iters);
        
    err  = infnorm(y1-y2);
    maxy = infnorm(y1);
    if err >= 10 * eps(maxy)
        keyboard;
    end
    
    y1 = BlochTorreyAction(z,h,D,f,gsize(1:3),iters,true);
    y1 = reshape(y1,size(x));
    y2 = BlochTorreyAction_brute(x,h,D,conj(f),iters);
    
    err  = infnorm(y1-y2);
    maxy = infnorm(y1);
    if err >= 10 * eps(maxy)
        keyboard;
    end
end

display_toc_time(toc(test_time),label);

end
