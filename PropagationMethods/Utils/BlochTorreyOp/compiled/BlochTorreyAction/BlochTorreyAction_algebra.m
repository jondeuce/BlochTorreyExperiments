function [] = BlochTorreyAction_algebra()
%BLOCHTORREYACTION_ALGEBRA

syms D h x y z real
xr = sym('xr(x,y,z)','real');
xi = sym('xi(x,y,z)','real');
Dr = sym('Dr(x,y,z)','real');
Di = sym('Di(x,y,z)','real');
Gr = sym('Gr(x,y,z)','real');
Gi = sym('Gi(x,y,z)','real');

xr = set_assumptions(xr,x,y,z);
xi = set_assumptions(xi,x,y,z);
Dr = set_assumptions(Dr,x,y,z);
Di = set_assumptions(Di,x,y,z);
Gr = set_assumptions(Gr,x,y,z);
Gi = set_assumptions(Gi,x,y,z);

fwd_grad = @(f) [subs(f,x,x+1) - f; subs(f,y,y+1) - f; subs(f,z,z+1) - f]/h;
bwd_grad = @(f) [f - subs(f,x,x-1); f - subs(f,y,y-1); f - subs(f,z,z-1)]/h;
fwd_div  = @(f) (subs(f(1),x,x+1) - f(1) + subs(f(2),y,y+1) - f(2) + subs(f(3),z,z+1) - f(3))/h;
bwd_div  = @(f) (f(1) - subs(f(1),x,x-1) + f(2) - subs(f(2),y,y-1) + f(3) - subs(f(3),z,z-1))/h;

cplx = @(x,y) x+1i*y;
real = @(x) (x+conj(x))/2;
imag = @(x) (x-conj(x))/2i;
xc = cplx(xr,xi);
Dc = cplx(Dr,Di);
Gc = cplx(Gr,Gi);

cleanexpr = @(exp) simplify(expand(exp));

BTop_Const_D = cleanexpr( D*bwd_div(fwd_grad(xc)) - Gc*xc );
BTop_Const_D = collect_recurse( BTop_Const_D, D, 1i );
disp(to_C_str(BTop_Const_D));

BTop_Array_D = cleanexpr( (bwd_div(Dr*fwd_grad(xc)) + fwd_div(Dr*bwd_grad(xc)))/2 - Gc*xc );
BTop_Array_D = collect_recurse( BTop_Array_D, sym(1i) );
BTop_Array_D_Real = collect_recurse( simplify(expand(real(BTop_Array_D))), sym(1/2) );
BTop_Array_D_Imag = collect_recurse( simplify(expand(imag(BTop_Array_D))), sym(1/2) );

BTop_Array_D_Real_Factors = factor(BTop_Array_D_Real);
BTop_Array_D_Real_Factors(2) = collect_recurse(BTop_Array_D_Real_Factors(2),Dr,subs(Dr,x,x+1),subs(Dr,x,x-1),subs(Dr,y,y+1),subs(Dr,y,y-1),subs(Dr,z,z+1),subs(Dr,z,z-1));
disp(to_C_str(BTop_Array_D_Real_Factors));

BTop_Array_D_Imag_Factors = -factor(BTop_Array_D_Imag);
BTop_Array_D_Imag_Factors(2) = collect_recurse(BTop_Array_D_Imag_Factors(2),Dr,subs(Dr,x,x+1),subs(Dr,x,x-1),subs(Dr,y,y+1),subs(Dr,y,y-1),subs(Dr,z,z+1),subs(Dr,z,z-1));
disp(to_C_str(BTop_Array_D_Imag_Factors));

end

function var = set_assumptions(var,x,y,z)
assumeAlso(var,'real');
assumeAlso(subs(var,x,x-1),'real');
assumeAlso(subs(var,x,x+1),'real');
assumeAlso(subs(var,y,y-1),'real');
assumeAlso(subs(var,y,y+1),'real');
assumeAlso(subs(var,z,z-1),'real');
assumeAlso(subs(var,z,z+1),'real');
end

function expr = collect_recurse(expr, varargin)

if length(varargin) <= 1
    expr = collect(expr, varargin{1});
else
    expr = collect(expr, varargin{1});
    expr = collect_recurse(expr, varargin{2:end});
end

end

function str = to_C_str(expr)

str = evalc('disp(expr)'); % get output of 'disp(expr)'

str = strrep(str,'(x, y, z)','[l]');
str = strrep(str,'(x - 1, y, z)','[l-1]');
str = strrep(str,'(x + 1, y, z)','[l+1]');
str = strrep(str,'(x, y - 1, z)','[jl]');
str = strrep(str,'(x, y + 1, z)','[jr]');
str = strrep(str,'(x, y, z - 1)','[kl]');
str = strrep(str,'(x, y, z + 1)','[kr]');

end