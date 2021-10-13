function [x] = MGPCG(b, Mask, iters, d, h)
%MGPCG   [x] = MGPCG(b, Mask, iters, d, h)
%   Detailed explanation goes here
 
    if nargin < 2 || isempty(Mask)
        Mask = true(size(b));
    end
    
    if nargin < 4 || isempty(d)
        d = fmg_max_depth(Mask);
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
        iters.pre = 5;
    end
    
    if ~isfield(iters, 'post')
        iters.post = 5;
    end
    
    maxIter = 50;
    v    = 1;
    tol  = 1e-4;
    
    b    = b .* Mask;
    
    %nb  = sqrt(b(:)' * b(:));
    nb   = infnorm(b);
    tolb = tol * nb;
    
    afun = @(x) A(x, Mask, h);
    mfun = @(x) M(x, Mask, iters, h, d);
    
    x   = zeros(size(b),'like',b);
    r   = b - afun(x);
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
        
%         nr = sqrt(rho1);
%         if nr <= tolb
%             r  = b - afun(x);
            %nr= sqrt(r(:)' * r(:));
            nr = infnorm(r);
            %if nr<= tolb
            if nr <= tol
                pv(ii, nr/nb);
                break
            end
%         end
        rho  = rho1;
        pv(ii, nr/nb);
    end
    
end

function print_progress(ii,res,maxIter,v)

    if v && (ii-1 <= 10 || ~mod(ii-1, v))
        fprintf('Iter: %3d/%3d, Res: %.8f\n', ii, maxIter, res);
    end
    
end

function [x] = A(x, Mask, h)

    D = reshape(-10-sin(linspace(0,pi,numel(x)).'), size(x));

%     x = (1/h^2) * (-6.*x+circshift(x,1,1)+circshift(x,-1,1)+circshift(x,1,2)+circshift(x,-1,2)+circshift(x,1,3)+circshift(x,-1,3));
%     x = force_per(x);
%     x([1,end],:,:) = 0;
%     x(:,[1,end],:) = 0;
%     x(:,:,[1,end]) = 0;
    x = D.*x + lap(x, Mask, h);
%     x = lap(x, Mask, h);
    
end

function [x] = M(b, G, it, h, d)

%     x = b;  return;
    x = zeros(size(G),'like',b);
%     x = v_cycle(x, b, G, it, h, d);
    x = v_cycle_per(x, b, G, it, h, d);
    
end

function [x] = v_cycle(x, b, G, it, h, d)
    
    if d > 1
        x = gs_forward_mex(x, b, G, it.pre, h);
        
        r = b - A(x, G, h);
        [r, G2] = restrict_mex(r, G);
        
        v = zeros(size(r),'like',r);
        v = v_cycle(v, r, G2, it, 2*h, d-1);
        x = x + prolong_mex(v, G2, G);
        
        x = gs_backward_mex(x, b, G, it.post, h);
    else
        x = gs_forward_mex(x, b, G, 500, h);
        x = gs_backward_mex(x, b, G, 500, h);
    end
    
end

function [x] = jac_relax(x, b, maxIter, h)

    w1 = 2/3;
    w2 = 1-w1;
    R  = @(x) (1/h^2) * (circshift(x,1,1)+circshift(x,-1,1)+circshift(x,1,2)+circshift(x,-1,2)+circshift(x,1,3)+circshift(x,-1,3));
    invD = 1./( reshape(-10-sin(linspace(0,pi,numel(x)).'), size(x)) - 6/h^2 );
    invD = @(x) invD.*x;
%     invD = -h^2/6;
    
    for k = 1:maxIter
%         x = (w1*invD)*(b-R(x)) + w2*x;
        x = w1*invD((b-R(x))) + w2*x;
%         x([1,end],:,:) = 0;
%         x(:,[1,end],:) = 0;
%         x(:,:,[1,end]) = 0;
%         x = force_per(x);
    end
    
end

function [x] = force_per(x)
    x([1,end],:,:) = repmat(0.5*(x(1,:,:)+x(end,:,:)),2,1,1);
    x(:,[1,end],:) = repmat(0.5*(x(:,1,:)+x(:,end,:)),1,2,1);
    x(:,:,[1,end]) = repmat(0.5*(x(:,:,1)+x(:,:,end)),1,1,2);
end

function [x] = v_cycle_per(x, b, G, it, h, d)
    
    if d > 1
        x = jac_relax(x, b, it.pre, h);
%         x = gs_forward_mex(x, b, G, it.pre, h);
        
        r = b - A(x, G, h);
        r = mgpcg_restrict(r);
%         r = force_per(mgpcg_restrict(r));
        G2= true(size(r));
        
        v = zeros(size(r),'like',r);
        v = v_cycle_per(v, r, G2, it, 2*h, d-1);
        x = x + mgpcg_prolong(v);
%         x = x + force_per(mgpcg_prolong(v));
        
        x = jac_relax(x, b, it.post, h);
%         x = gs_backward_mex(x, b, G, it.post, h);
    else
        x = jac_relax(x, b, it.pre+it.post, h);
%         x = gs_forward_mex(x, b, G, numel(b), h);
%         x = gs_backward_mex(x, b, G, numel(b), h);
    end
    
end

function [d] = fmg_max_depth(G)
    
    mSize = size(crop2mask(G));
    d = max(min(floor(log2(mSize))) - 2, 1);
    
end
