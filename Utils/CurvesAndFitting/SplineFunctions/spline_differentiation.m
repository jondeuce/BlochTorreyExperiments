function dy = spline_differentiation( x, y, n_int, num_splines )
%SPLINE_DIFFERENTIATION This function aims to test the idea that, given a
%noisey signal y=f(x), a good approximation of the derivative in the sense
%of where the function is 'trying to go', is to integrate y 'n_int' times,
%fit a smooth function (in this case, a cubic spline) to the (presumably
%smoother) integrated y, and differentiate this spline 'n_int' times and
%evaluate at 'x' to receive the desired derivatives
% 
%   'x' is the independent variable
%   'y' is the dependent (noisey) signal
%   'n_int' is the number of times to integrate
%   'num_splines' is the number of splines to use when fitting the
%   integral. will assume uniform spacing, or can also be a set of indexes,
%       e.g. points [1, 37, 92, 120], data of length 120

if isscalar(num_splines)
    breaks=round(linspace(1,length(x),num_splines));
else
    breaks=num_splines;
    if breaks(1)~=1; breaks=[1, breaks]; end
    if breaks(end)~=length(x); breaks=[breaks, length(x)]; end
end

x=x(:); y=y(:);

iy=num_int(x,y,n_int);
cs_int=splinefit(x,y,x(breaks));
cs=differentiate_spline(cs_int,n_int);

dy=ppval(cs,x);

figure, hold on, grid on
h1=plot(x,y,'r--');
h2=plot(x,iy,'b--');
h3=plot(x,dy,'g--');

int_sign='\int';
str = ['$$' repmat(int_sign,[1,n_int]) ' y dx$$'];
legend([h1;h2;h3],{'y',str,'dy'},'interpreter','latex');

end

function iy = num_int(x,y,n_int)

iy = cumtrapz(x,y);

if n_int>1
    iy = iy - mean(iy);
    iy = num_int(x,iy,n_int-1);
end

end

function cs_df = differentiate_spline(cs,n)

if nargin<2; n=1; end;

[breaks,coefs,l,k,d] = unmkpp(cs);
cs_df = mkpp(breaks,repmat(k-1:-1:1,[d*l,1]).*coefs(:,1:k-1),d);

if n>1
    cs_df = differentiate_spline(cs_df,n-1);
end

end

function cs_int = integrate_spline(cs,n)

if nargin<2; n=1; end;

[breaks,coefs,l,k,d] = unmkpp(cs);
cs_int = mkpp( breaks,...
    repmat( [1./(k:-1:1), 0],[d*l,1]) .* [coefs, zeros(size(coefs,1),1)],d);

if n>1
    cs_int = integrate_spline(cs_int,n-1);
end

end