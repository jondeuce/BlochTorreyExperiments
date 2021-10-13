function [x] = fmg_relax_jac(x, A, b, maxIter, h)
% FMG_RELAX_JAC Relaxation step for FMG using the (weighted) Jacobi method
% 
% Creating matfun for use with FMG_RELAX_JAC, define the following methods:
% diag, issparse.

    if nargin < 5 || isempty(h)
        h = 1;
    end
    if isscalar(h)
        h = [h, h, h];
    end
    if nargin < 4, maxIter = 250; end
    if isempty(x)
        x = zeros(size(b),'like',b);
    end
    
    % Jacobi method: x <- w1*inv(D)*(b-R*x) + w2*x
    %                  =  w1*inv(D)*b - (w1*inv(D)*R - w2)*x
    %                 :=  D2 - R2*x
    
    % Parameters
    w1 = 1;
    w2 = 1-w1;
    n  = numel(b);
    
    % Extract diagonal
    D  = diag(A);
    if issparse(D)
        D  = full(D);
    end
    
    % Create R matrix
    if issparse(A)
        R  = A - spdiags(D,0,n,n);
    elseif ismatfun(A)
        DI = matfun(@(x)D.*x,size(A));
        R  = A - DI;
    else
        R  = A;
        R(1:n+1:n^2) = 0; % R = A - diag(diag(A))
    end
    
    % Create D2 and R2 matrices
    D  = w1./D;
    if issparse(A)
        R  = spdiags(D,0,n,n) * R;
        R  = R - spdiags(w2.*ones(n,1),0,n,n);
    elseif ismatfun(A)
        DI = matfun(@(x)D.*x,size(A));
        W2 = matfun(@(x)w2.*x,size(A));
        R  = DI * R - W2;
    else
        R  = bsxfun(@times,D,R); % D*R
        R(1:n+1:n^2) = R(1:n+1:n^2) - w2; % R = R - w2*I
    end
    D  = D.*b;
    
    % Iterate Jacobi method
    for ii = 1:maxIter
        x  = D - R*x;
    end
    
    if ~isequal(size(x),size(b))
        x  = reshape(x,size(b));
    end
    
end

