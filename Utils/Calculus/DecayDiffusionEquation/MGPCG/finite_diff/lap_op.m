function [delOp, idelOp] = lap_op(mSize, prec)
%LAP_OP Discrete Laplacian Convolution Kernel

    if nargin < 2, prec = 'single'; end
    
    % Laplace Kernel
    kernel(:,:,1) = [0 0 0; 0  1  0; 0 0 0];
    kernel(:,:,2) = [0 1 0; 1 -6  1; 0 1 0];
    kernel(:,:,3) = [0 0 0; 0  1  0; 0 0 0];
    
    delOp = psf2otf(kernel, mSize);

    if strcmpi(prec, 'single')
        delOp = single(delOp);
    end
    
    if nargout > 1
        idelOp = 1 ./ delOp;
        idelOp(delOp == 0) = 0;
    end
    
end
