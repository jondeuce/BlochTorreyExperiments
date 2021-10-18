function test_fmg_diffuse
%TEST_FMG_DIFFUSE Tests the function fmg_diffuse(x)

n = 100;
s = num2str(n);
main_test([s,' complex double'],n,50,[1,3]);

end

function main_test(label,n,varargin)

test_time = tic;

for ii = 1:n
    x  = randnc(1+randi(varargin{:}),'double');
    f  = randnc(size(x),'double');
    h  = rand;
    D  = rand;
    c  = 2*rand-1;
    y1 = fmg_diffuse(x,h,D,f,c);
    y2 = c * fmg_diffuse_brute(x,h,D,f);
    
    err  = infnorm(y1-y2);
    maxy = infnorm(y1);
    try
        assert( err < 10 * eps(maxy) );
    catch me
        keyboard;
        rethrow(me);
    end
end

display_toc_time(toc(test_time)/n,label);

end

function y = fmg_diffuse_brute(x, h, D, f)
% y = D*lap(x,h)-f*x

% y = lap(x,1)
y = -6 * x;
y = y + circshift(x, 1, 1);
y = y + circshift(x,-1, 1);
y = y + circshift(x, 1, 2);
y = y + circshift(x,-1, 2);
y = y + circshift(x, 1, 3);
y = y + circshift(x,-1, 3);

% y = D/h^2*lap(x,1)
K = D/h^2;
y = K * y;

% y = D/h^2*lap(x,1) - f
y = y - f.*x;

end