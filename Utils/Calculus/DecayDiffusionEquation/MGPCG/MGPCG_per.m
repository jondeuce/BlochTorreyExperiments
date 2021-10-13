function [x] = MGPCG_per(b, f, iters, d, h)
%MGPCG_PER [x] = MGPCG_per(b, f, iters, d, h)
 
    if nargin < 2 || isempty(f)
        f = zeros(size(b),'like',b);
    end
    
    if nargin < 4 || isempty(d)
        d = fmg_max_depth(b);
    end
    
    if nargin < 5
        h = 1;
    end
    
    if nargin < 3 || isempty(iters)
        iters = [];
    end
    
    if ~isstruct(iters) && ~isempty(iters)
        iters = struct('pre', iters, 'post', iters);
    end
    
    if ~isfield(iters, 'pre')
        iters.pre = 2;
    end
    
    if ~isfield(iters, 'post')
        iters.post = 2;
    end
    
    maxIter = 10;
    v    = 1;
    tol  = 1e-15;
    
    %nb  = sqrt(b(:)' * b(:));
    %tolb= tol * nb;
    %nb  = infnorm(b);
    nb   = maxabs(b);
    
    afun = @(x) A(x, f, h);
    mfun = @(x) M(x, f, iters, h, d);
    
    x   = zeros(size(b),'like',b);
    %r  = b - afun(x);
    r   = b;
    z   = mfun(r);
    p   = z;
    rho = p(:)' * r(:);
    
    pv   = @(ii,res) print_progress(ii,res,maxIter,v);
    
    for ii = 1:maxIter        
        
        z = afun(p);
        alpha = rho / (p(:)' * z(:));
        x = x + alpha*p;
        r = r - alpha*z;
        
        z = mfun(r);
        rho1 = z(:)' * r(:);
        p = z + rho1/rho*p;
        
        %nr= infnorm(r);
        nr = maxabs(r);
        if nr <= tol
            pv(ii, nr/nb);
            break
        end
        
        rho  = rho1;
        pv(ii, nr/nb);
        
    end
    
end

function print_progress(ii,res,maxIter,v)

    if v && (ii-1 <= 10 || ~mod(ii-1, v))
        fprintf('Iter: %3d/%3d, Res: %.8f\n', ii, maxIter, res);
    end
    
end

function [x] = A(x, f, h)

%     x = (1/h^2) * ( -6.*x + circshift(x,1,1) + circshift(x,-1,1) + ...
%                             circshift(x,1,2) + circshift(x,-1,2) + ...
%                             circshift(x,1,3) + circshift(x,-1,3) ) - f.*x;
    x = fmg_diffuse(x,h,1,f);
    
end

function [x] = M(b, f, it, h, d)

%     x = b;  return;
    x = zeros(size(b),'like',b);
    x = v_cycle_per(x, b, f, it, h, d);
    
end

function [x] = jac_relax(x, b, f, maxIter, h)

    w1 = 2/3;
    w2 = 1-w1;
    
%     R = @(x) (1/h^2) * (circshift(x,1,1)+circshift(x,-1,1)+circshift(x,1,2)+circshift(x,-1,2)+circshift(x,1,3)+circshift(x,-1,3));
%     invD = -1./(f+6/h^2);
%     invD = @(x) invD.*x;
    
    R = @(x) fmg_lap_per(x,h) + (6/h^2)*x;
    invD = -1./(f+6/h^2);
    invD = @(x) invD.*x;
    
    for k = 1:maxIter
        x = w1*invD((b-R(x))) + w2*x;
    end
    
end

function [x] = v_cycle_per(x, b, f, it, h, d)
    
    if d > 1
        %x= jac_relax(x, b, f, it.pre, h);
        if isreal(x), x = complex(x); end
        if isreal(b), b = complex(b); end
        if isreal(f), f = complex(f); end
        x = sor_diffuse(b, x, 1.0, h, 1.0, f, 1, it.pre, 0.0);
        
        r  = b - A(x, f, h);
        r  = mgpcg_restrict(r);
        f2 = mgpcg_restrict(f);
        
        v = zeros(size(r),'like',r);
        v = v_cycle_per(v, r, f2, it, 2*h, d-1);
        x = x + mgpcg_prolong(v);
        
        %x= jac_relax(x, b, f, it.post, h);
        if isreal(x), x = complex(x); end
        if isreal(b), b = complex(b); end
        if isreal(f), f = complex(f); end
        x = sor_diffuse(b, x, 1.0, h, 1.0, f, 1, it.post, 1.0);
    else
        %x= jac_relax(x, b, f, it.pre + it.post, h);
        if isreal(x), x = complex(x); end
        if isreal(b), b = complex(b); end
        if isreal(f), f = complex(f); end
        x = sor_diffuse(b, x, 1.0, h, 1.0, f, 1, numel(b), 0.0);
        x = sor_diffuse(b, x, 1.0, h, 1.0, f, 1, numel(b), 1.0);
    end
    
end

function [d] = fmg_max_depth(x)
    
    d = max(min(floor(log2(size(x)))) - 2, 1);
    
end

%{
function [x] = force_per(x)
    x([1,end],:,:) = repmat(0.5*(x(1,:,:)+x(end,:,:)),2,1,1);
    x(:,[1,end],:) = repmat(0.5*(x(:,1,:)+x(:,end,:)),1,2,1);
    x(:,:,[1,end]) = repmat(0.5*(x(:,:,1)+x(:,:,end)),1,1,2);
end
%}