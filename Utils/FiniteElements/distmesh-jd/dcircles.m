function d=dcircles(p,xc,yc,r,issigned)

if nargin < 5; issigned = true; end

if issigned
    dcircle_fun = @dcircle;
else
    dcircle_fun = @(p,xc,yc,r) abs(dcircle(p,xc,yc,r));
end

d = dcircle_fun(p,xc(1),yc(1),r(1));
for ii = 2:length(r)
    d = dunion(d, dcircle_fun(p,xc(ii),yc(ii),r(ii)));
end

end